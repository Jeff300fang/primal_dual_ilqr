from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import jit, lax, scipy, vmap
from jax.tree_util import register_pytree_node_class
# from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_associative_scan import associative_scan_cache_acp_jax, associative_scan_use_cache_cp_jax
from typing import NamedTuple
import math
from functools import partial
# -----------------------------
# Cache container (PyTree-friendly)
# -----------------------------
class ACPScanCache(NamedTuple):
    Ar:  jnp.ndarray   # (2L, T, n, n)
    Al:  jnp.ndarray
    ArC: jnp.ndarray
    AlP: jnp.ndarray
    Cn:  jnp.ndarray
    Pn:  jnp.ndarray

@dataclass
class ADMMConfig:
    rho_update_frequency: int = 25
    max_iterations: int = 600
    eps_abs: float = 1e-2
    eps_rel: float = 1e-2
    condense_block_size: int = 1

@register_pytree_node_class
@dataclass
class ADMMWarmStart:
    w: jnp.ndarray          # (T+1, m)
    y: jnp.ndarray          # (T+1, m)
    rho: jnp.ndarray  

    def tree_flatten(self):
        children = (self.w, self.y, self.rho)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# -----------------------------
# JAX combine for full TVLQR element (for reference comparison)
# -----------------------------
def fn_full(next_elem, prev_elem):
    n = prev_elem.shape[-1]

    A_l = prev_elem[0:n, :]
    c_l = prev_elem[n, :]
    C_l = prev_elem[n + 1 : 2 * n + 1, :]
    p_l = prev_elem[2 * n + 1, :]
    P_l = prev_elem[-n:, :]

    A_r = next_elem[0:n, :]
    c_r = next_elem[n, :]
    C_r = next_elem[n + 1 : 2 * n + 1, :]
    p_r = next_elem[2 * n + 1, :]
    P_r = next_elem[-n:, :]

    Ar = A_r @ jnp.linalg.inv(jnp.eye(n, dtype=prev_elem.dtype) + C_l @ P_r)
    Al = A_l.T @ jnp.linalg.inv(jnp.eye(n, dtype=prev_elem.dtype) + P_r @ C_l)

    A_new = Ar @ A_l
    c_new = Ar @ (c_l - C_l @ p_r) + c_r
    C_new = Ar @ C_l @ A_r.T + C_r
    p_new = Al @ (p_r + P_r @ c_l) + p_l
    P_new = Al @ P_r @ A_l + P_l

    return jnp.concatenate(
        [A_new, c_new.reshape(1, n), C_new, p_new.reshape(1, n), P_new],
        axis=0,
    )


# -----------------------------
# Helpers (static-shape masking)
# -----------------------------
def _shift_down(x, step):
    pad = jnp.zeros((step,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([pad, x[:-step]], axis=0)

def _mask_upsweep(T, step):
    t = jnp.arange(T)
    period = 2 * step
    return (t % period) == (period - 1)

def _mask_downsweep(T, step):
    t = jnp.arange(T)
    period = 2 * step
    return (t >= (3 * step - 1)) & ((t % period) == (step - 1))

def _masked_write_level(cache, level, vals, mask):
    # cache[level, t] = vals[t] if mask[t] else 0
    m = mask.astype(vals.dtype)[:, None, None]
    return cache.at[level].set(vals * m)

def _combine_acp_all(next_block, prev_block, n):
    """
    next_block, prev_block: (T, 3n, n)
    Returns:
      combined: (T, 3n, n)
      Ar, Al, ArC, AlP, C_new, P_new: (T, n, n)
    """
    dtype = prev_block.dtype
    I = jnp.eye(n, dtype=dtype)[None, :, :]  # (1,n,n) for broadcasting

    A_l = prev_block[:, 0:n, :]
    C_l = prev_block[:, n:2*n, :]
    P_l = prev_block[:, 2*n:3*n, :]

    A_r = next_block[:, 0:n, :]
    C_r = next_block[:, n:2*n, :]
    P_r = next_block[:, 2*n:3*n, :]

    inv1 = jnp.linalg.inv(I + C_l @ P_r)
    inv2 = jnp.linalg.inv(I + P_r @ C_l)

    Ar = A_r @ inv1
    Al = jnp.swapaxes(A_l, -1, -2) @ inv2

    ArC = Ar @ C_l
    AlP = Al @ P_r

    A_new = Ar @ A_l
    C_new = ArC @ jnp.swapaxes(A_r, -1, -2) + C_r
    P_new = AlP @ A_l + P_l

    combined = jnp.concatenate([A_new, C_new, P_new], axis=1)
    return combined, Ar, Al, ArC, AlP, C_new, P_new


# -----------------------------
# JAX: cache-producing scan over (A,C,P)
# -----------------------------
@partial(jax.jit, static_argnums=(1, 2, 3))
def associative_scan_cache_acp_jax(elems_acp, T: int, n: int, reverse: bool = False):
    dtype = elems_acp.dtype
    L = int(math.ceil(math.log2(max(T, 1))))

    Ar  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Al  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    ArC = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    AlP = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Cn  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Pn  = jnp.zeros((2 * L, T, n, n), dtype=dtype)

    x = elems_acp[::-1] if reverse else elems_acp
    out = x

    # Upsweep
    step = 1
    level = 0
    while step < T and level < L:
        mask = _mask_upsweep(T, step)
        next_block = _shift_down(out, step)
        prev_block = out

        combined, aR, aL, aRC, aLP, cN, pN = _combine_acp_all(next_block, prev_block, n)

        m = mask[:, None, None]
        out = jnp.where(m, combined, out)

        Ar  = _masked_write_level(Ar,  level, aR,  mask)
        Al  = _masked_write_level(Al,  level, aL,  mask)
        ArC = _masked_write_level(ArC, level, aRC, mask)
        AlP = _masked_write_level(AlP, level, aLP, mask)
        Cn  = _masked_write_level(Cn,  level, cN,  mask)
        Pn  = _masked_write_level(Pn,  level, pN,  mask)

        step *= 2
        level += 1

    # Downsweep-style fill
    step //= 4
    level2 = 0
    while step >= 1 and level2 < L:
        mask = _mask_downsweep(T, step)
        next_block = _shift_down(out, step)
        prev_block = out

        combined, aR, aL, aRC, aLP, cN, pN = _combine_acp_all(next_block, prev_block, n)

        m = mask[:, None, None]
        out = jnp.where(m, combined, out)

        lvl = L + level2
        Ar  = _masked_write_level(Ar,  lvl, aR,  mask)
        Al  = _masked_write_level(Al,  lvl, aL,  mask)
        ArC = _masked_write_level(ArC, lvl, aRC, mask)
        AlP = _masked_write_level(AlP, lvl, aLP, mask)
        Cn  = _masked_write_level(Cn,  lvl, cN,  mask)
        Pn  = _masked_write_level(Pn,  lvl, pN,  mask)

        step //= 2
        level2 += 1

    if reverse:
        out = out[::-1]

    return out, ACPScanCache(Ar=Ar, Al=Al, ArC=ArC, AlP=AlP, Cn=Cn, Pn=Pn)


# -----------------------------
# JAX: propagate (c,p) using caches (no inverses)
# -----------------------------
@partial(jax.jit, static_argnums=(2, 4))
def associative_scan_use_cache_cp_jax(c, p, T: int, cache: ACPScanCache, reverse: bool = False):
    Ar, Al, ArC, AlP = cache.Ar, cache.Al, cache.ArC, cache.AlP
    L = Ar.shape[0] // 2

    c_out = c[::-1] if reverse else c
    p_out = p[::-1] if reverse else p

    # Upsweep
    step = 1
    level = 0
    while step < T and level < L:
        mask = _mask_upsweep(T, step)
        m = mask.astype(c_out.dtype)[:, None]

        c_l = c_out
        p_l = p_out
        c_r = _shift_down(c_out, step)
        p_r = _shift_down(p_out, step)

        c_new = (Ar[level]  @ c_l[..., None]).squeeze(-1) - (ArC[level] @ p_r[..., None]).squeeze(-1) + c_r
        p_new = (Al[level]  @ p_r[..., None]).squeeze(-1) + (AlP[level] @ c_l[..., None]).squeeze(-1) + p_l

        c_out = c_out + m * (c_new - c_out)
        p_out = p_out + m * (p_new - p_out)

        step *= 2
        level += 1

    # Downsweep
    step //= 4
    level2 = 0
    while step >= 1 and level2 < L:
        mask = _mask_downsweep(T, step)
        m = mask.astype(c_out.dtype)[:, None]
        lvl = L + level2

        c_l = c_out
        p_l = p_out
        c_r = _shift_down(c_out, step)
        p_r = _shift_down(p_out, step)

        c_new = (Ar[lvl]  @ c_l[..., None]).squeeze(-1) - (ArC[lvl] @ p_r[..., None]).squeeze(-1) + c_r
        p_new = (Al[lvl]  @ p_r[..., None]).squeeze(-1) + (AlP[lvl] @ c_l[..., None]).squeeze(-1) + p_l

        c_out = c_out + m * (c_new - c_out)
        p_out = p_out + m * (p_new - p_out)

        step //= 2
        level2 += 1

    if reverse:
        c_out = c_out[::-1]
        p_out = p_out[::-1]

    return c_out, p_out

@jit
def dual_lqr(X, P, p):
    """Dual LQR solve.

    Args:
      P: [T+1, n, n]   numpy array.
      p: [T+1, n]      numpy array.
      X: [T+1, n]      numpy array.

    Returns:
      V: [T+1, n] numpy array.
    """
    Tp1 = X.shape[0]
    return vmap(lambda t: P[t] @ X[t] + p[t])(jnp.arange(Tp1))

@jit
def rollout_gpu(K, k, x0, A, B, c):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
    T, _, n = K.shape

    def fn(prev, next):
        F = prev[:-1]
        f = prev[-1]
        G = next[:-1]
        g = next[-1]
        return jnp.concatenate([G @ F, (g + G @ f).reshape([1, n])])

    get_elem = lambda t: jnp.concatenate(
        [A[t] + B[t] @ K[t], (c[t] + B[t] @ k[t]).reshape([1, n])]
    )
    elems = vmap(get_elem)(jnp.arange(T))
    comp = lax.associative_scan(lambda l, r: vmap(fn)(l, r), elems)
    X = jnp.concatenate(
        [
            x0.reshape(1, n),
            vmap(lambda t: comp[t, :-1, :] @ x0 + comp[t, -1, :])(jnp.arange(T)),
        ]
    )

    U = vmap(lambda t: K[t] @ X[t] + k[t])(jnp.arange(T))

    return X, U

def admm_augment_xu(Q, q, R, r, M, C, D, w_bar, y_bar, rho):
    s_bar = w_bar - y_bar   
    # CtC: (T, nx, nx)
    CtC = jnp.einsum('tmi,tmj->tij', C, C)
    # DtD: (T, nu, nu)
    DtD = jnp.einsum('tmi,tmj->tij', D, D)
    # CtD: (T, nx, nu)
    CtD = jnp.einsum('tmi,tmj->tij', C, D)

    # C^T sbar: (T, nx)
    Ct_s = jnp.einsum('tmi,tm->ti', C, s_bar)
    # D^T sbar: (T, nu)
    Dt_s = jnp.einsum('tmi,tm->ti', D, s_bar)

    tilde_Q = Q + rho * CtC
    tilde_q = q - rho * Ct_s

    tilde_R = R + rho * DtD
    tilde_r = r - rho * Dt_s

    tilde_M = M + rho * CtD

    return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M

def admm_residuals(z, w, w_prev, y, rho, eps_abs=1e-2, eps_rel=1e-2):
    """
    z, w, w_prev, y: (T+1, m)
    Returns scalar norms and (optional) thresholds.
    """
    r = z - w                          # primal residual
    s = rho * (w - w_prev)             # dual residual (A=I)

    # Norms over all time/constraints
    r_norm = jnp.linalg.norm(r.reshape(-1), ord=2)
    s_norm = jnp.linalg.norm(s.reshape(-1), ord=2)

    # Stopping thresholds (Boyd et al., scaled form)
    n = r.size
    z_norm = jnp.linalg.norm(z.reshape(-1), ord=2)
    w_norm = jnp.linalg.norm(w.reshape(-1), ord=2)
    y_norm = jnp.linalg.norm(y.reshape(-1), ord=2)

    eps_pri = jnp.sqrt(n) * eps_abs + eps_rel * jnp.maximum(z_norm, w_norm)
    eps_dual = jnp.sqrt(n) * eps_abs + eps_rel * (rho * y_norm)

    return r_norm, s_norm, eps_pri, eps_dual

def adaptive_rho_update(rp_norm, rd_norm, rho,
                        mu=10.0, tau_inc=2.0, tau_dec=2.0,
                        rho_min=1e-2, rho_max=1e6):
    inc = rp_norm > mu * rd_norm
    dec = rd_norm > mu * rp_norm

    rho_inc = jnp.minimum(rho * tau_inc, rho_max)
    rho_dec = jnp.maximum(rho / tau_dec, rho_min)

    rho_new = jnp.where(inc, rho_inc,
                jnp.where(dec, rho_dec, rho))
    updated = rho_new != rho
    return rho_new, updated

def compute_Ginv(R, B, P):
    """
    Returns Ginv: (T, nu, nu)
    """
    T, nu, _ = R.shape
    Iu = jnp.eye(nu, dtype=R.dtype)

    def one(t):
        G = R[t] + B[t].T @ P[t+1] @ B[t]
        # Ginv = solve(G, I)
        return jnp.linalg.solve(G, Iu)

    return vmap(one)(jnp.arange(T))

def rho_update_y(rp_norm, rd_norm, rho, y):
    rho_new, updated = adaptive_rho_update(rp_norm, rd_norm, rho)
    y_new = lax.cond(
        updated,
        lambda _: (rho / rho_new) * y,
        lambda _: y,
        operand=None
    )
    return rho_new, y_new, updated

def generate_leaf(tilde_Q, tilde_R, tilde_M, A, B):
    T = tilde_Q.shape[0] - 1
    n = tilde_Q.shape[1]
    def chol_inv(t):
        f = scipy.linalg.cho_factor(tilde_R[t])
        m = tilde_R[t].shape[0]
        return scipy.linalg.cho_solve(f, jnp.eye(m))

    Rinv = vmap(chol_inv)(jnp.arange(T))
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))
    MRinv = vmap(lambda t: tilde_M[t] @ Rinv[t])(jnp.arange(T))

    elems = jnp.concatenate(
        [
            # The A matrices.
            jnp.concatenate(
                [
                    A - vmap(lambda t: BRinv[t] @ tilde_M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            # The C matrices.
            jnp.concatenate(
                [
                    vmap(lambda t: BRinv[t] @ B[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
            tilde_Q
            - jnp.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ tilde_M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
        ],
        axis=1,
    )

    return elems, BRinv, MRinv

def generate_leaf_bp(c, BRinv, MRinv, tilde_r, tilde_q, T, n):
    # c: (T, n)   where this is c[1:] in your caller
    # tilde_q: (T+1, n)
    # tilde_r: (T+1, nu) but only first T matter here

    c_stage = c - vmap(lambda t: BRinv[t] @ tilde_r[t])(jnp.arange(T))  # (T, n)
    c0 = jnp.concatenate([c_stage, jnp.zeros((1, n), dtype=c.dtype)], axis=0)  # (T+1, n)

    p_stage = vmap(lambda t: MRinv[t] @ tilde_r[t])(jnp.arange(T))  # (T, n)
    p0 = tilde_q - jnp.concatenate([p_stage, jnp.zeros((1, n), dtype=tilde_q.dtype)], axis=0)  # (T+1, n)

    return c0, p0

def get_k(tilde_R, tilde_r, B, P, p, b):
    T = B.shape[0]
    def one(t):
        BtP = B[t].T @ P[t + 1]
        G = tilde_R[t] + BtP @ B[t]
        h = B[t].T @ p[t + 1] + BtP @ b[t] + tilde_r[t]
        return scipy.linalg.solve(G, -h)
    return vmap(one)(jnp.arange(T))

def get_K(tilde_R, tilde_M, A, B, P):
    T = B.shape[0]  # NOT tilde_R.shape[0]
    def one(t):
        BtP = B[t].T @ P[t + 1]
        H  = BtP @ A[t] + tilde_M[t].T
        G  = tilde_R[t] + BtP @ B[t]
        return scipy.linalg.solve(G, -H)
    return vmap(one)(jnp.arange(T))

def constrained_solve(cfg: ADMMConfig, Q, q, R, r, M, A, B, c, C, D, f, w, y, rho):
    # --- one ADMM iteration ---
    def one_iter(carry):
        (it, tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, 
         x_bar, u_bar, y_bar, w_prev, rho, cache, BRinv, MRinv, P, _, K,
         _, _, _, _, _) = carry


        # -------- Solve unconstrained LQR subproblem --------
        T = Q.shape[0] - 1
        n = Q.shape[1]
        c0, p0 = generate_leaf_bp(c[1:], BRinv, MRinv, tilde_r, tilde_q, T, n)
        b, p = associative_scan_use_cache_cp_jax(c0, p0, T + 1, cache, reverse=True)
        k = get_k(tilde_R, tilde_r, B, P, p, c[1:])

        x_bar, u_stage = rollout_gpu(K, k, c[0], A, B, c[1:])
        u_bar = jnp.pad(u_stage, ((0, 1), (0, 0)))  # (T+1, nu)

        # z = Cx + Du
        z_bar = (
            jnp.einsum('tmi,ti->tm', C, x_bar) +
            jnp.einsum('tmi,ti->tm', D, u_bar)
        )

        # -------- Project onto constraint set -------- 
        w_new = jnp.minimum(z_bar + y_bar, f)

        # -------- Dual update (scaled form) -------- 
        y_new = y_bar + (z_bar - w_new)


        # -------- Termination + Rho/Cache Update -------- 
        # Residual norms and tolerances
        rp_norm, rd_norm, eps_pri, eps_dual = admm_residuals(
            z_bar, w_new, w_prev, y_new, rho,
            eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel
        )

        # Convergence check
        converged = jnp.logical_and(rp_norm <= eps_pri, rd_norm <= eps_dual)
        # Adaptive rho (gated)
        do_rho_update = (it % cfg.rho_update_frequency) == 0

        def update_fn(_):
            rho_upd, y_upd, updated = rho_update_y(
                rp_norm, rd_norm,
                rho, y_new
            )
            return rho_upd, y_upd, updated

        def no_update_fn(_):
            return rho, y_new, jnp.array(False)

        def cache_update(_):
            tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M = admm_augment_xu(
                Q, q, R, r, M, C, D, w_new, y_new, rho_new
            )
            Tp1 = Q.shape[0]   # = T+1
            T   = Tp1 - 1
            n   = Q.shape[1]

            elems_acp, BRinv, MRinv = generate_leaf(tilde_Q, tilde_R, tilde_M, A, B)
            out_acp, cache = associative_scan_cache_acp_jax(elems_acp, Tp1, n, reverse=True)

            P = out_acp[:, -n:, :]
            K = get_K(tilde_R, tilde_M, A, B, P)
            return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache, BRinv, MRinv, P, K

        def no_cache_update(_):
            return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache, BRinv, MRinv, P, K

        # jax.debug.print(
        #     "ADMM: it={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e})",
        #     it, rho, rp_norm, eps_pri, rd_norm, eps_dual
        # )
        rho_new, y_new, rho_updated = lax.cond(do_rho_update, update_fn, no_update_fn, operand=None)
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache_new, BRinv, MRinv, P, K = lax.cond(rho_updated, cache_update, no_cache_update, operand=None)

        return (it + 1, tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, x_bar, u_bar, y_new, w_new,
                rho_new, cache_new, BRinv, MRinv, P, p, K,
                rp_norm, rd_norm, eps_pri, eps_dual, converged)

    # --- loop condition: keep going until max_iters OR converged ---
    def cond_fun(carry):
        it = carry[0]
        converged = carry[-1]
        return jnp.logical_and(it < cfg.max_iterations, jnp.logical_not(converged))
    
    T = A.shape[0]
    n = Q.shape[1]
    nx = Q.shape[-1]
    nu = R.shape[-1]

    # Pad terminal for consistency with x_{T} term
    R = jnp.concatenate([R, jnp.zeros((1, nu, nu), dtype=R.dtype)], axis=0)
    r = jnp.concatenate([r, jnp.zeros((1, nu), dtype=r.dtype)], axis=0)
    M = jnp.concatenate([M, jnp.zeros((1, nx, nu), dtype=M.dtype)], axis=0)
    init_x = jnp.zeros((T + 1, nx), dtype=Q.dtype)
    init_u = jnp.zeros((T + 1, nu), dtype=Q.dtype)
    init_w = w
    init_y = y
    rho0 = rho
    p_init = jnp.zeros((T + 1, nx), dtype=Q.dtype)
    tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M = admm_augment_xu(
        Q, q, R, r, M, C, D, init_w, init_y, rho0
    )
    elems_acp, BRinv, MRinv = generate_leaf(tilde_Q, tilde_R, tilde_M, A, B)
    out_acp, cache = associative_scan_cache_acp_jax(elems_acp, T + 1, n, reverse=True)
    P = out_acp[:, -n:, :]
    K = get_K(tilde_R, tilde_M, A, B, P)
    init = (
        jnp.array(1, dtype=jnp.int32),  # it
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M,
        init_x, init_u, init_y, init_w,                          
        jnp.array(rho0, dtype=Q.dtype),  # rho
        cache, BRinv, MRinv, P, p_init, K,
        jnp.array(jnp.inf, dtype=Q.dtype),  # rp_norm
        jnp.array(jnp.inf, dtype=Q.dtype),  # rd_norm
        jnp.array(jnp.inf, dtype=Q.dtype),  # eps_pri
        jnp.array(jnp.inf, dtype=Q.dtype),  # eps_dual
        jnp.array(False)                    # converged
    )

    out = jax.lax.while_loop(cond_fun, one_iter, init)

    it, _, _, _, _, _, x_bar, u_bar, y_bar, w_bar, rho_final, _, _, _, P_final, p_final, _, rp_norm, rd_norm, eps_pri, eps_dual, converged = out

    v = dual_lqr(x_bar, P_final, p_final)
    jax.debug.print(
        "ADMM done: Total Iterations={} converged={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e})",
        it - 1, converged, rho_final, rp_norm, eps_pri, rd_norm, eps_dual
    )

    return x_bar, u_bar[:-1], v, w_bar, y_bar, rho_final