from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import jit, lax, scipy, vmap
from jax.tree_util import register_pytree_node_class
from typing import NamedTuple
import math
from functools import partial
import time

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

@register_pytree_node_class
@dataclass(frozen=True)
class ADMMConfig:
    rho_update_frequency: int = 25
    max_iterations: int = 400
    eps_abs: float = 1e-2
    eps_rel: float = 1e-2
    condense_block_size: int = 1
    rho_max: int = 1e5

    def tree_flatten(self):
        children = (self.rho_update_frequency, self.max_iterations,
                    self.eps_abs, self.eps_rel,
                    self.condense_block_size, self.rho_max)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

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
# Modified Ruiz-style diagonal scaling (OSQP-inspired)
# -----------------------------
@register_pytree_node_class
@dataclass
class RuizScaling:
    # Variable scalings
    Sx: jnp.ndarray      # (nx,)
    Su: jnp.ndarray      # (nu,)
    # Constraint-row scalings (only for inequality rows z = Cx + Du)
    E:  jnp.ndarray      # (m,)
    # Inverses
    Sx_inv: jnp.ndarray  # (nx,)
    Su_inv: jnp.ndarray  # (nu,)
    E_inv:  jnp.ndarray  # (m,)
    # Objective (cost) scaling: multiply whole objective by gamma (OSQP-like)
    gamma: jnp.ndarray   # scalar

    def tree_flatten(self):
        children = (self.Sx, self.Su, self.E, self.Sx_inv, self.Su_inv, self.E_inv, self.gamma)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

def _safe_inv(v, eps=1e-12):
    return 1.0 / jnp.maximum(v, eps)

def _safe_sqrt_inv(v, eps=1e-12):
    return jnp.sqrt(_safe_inv(v, eps))

def _col_inf_norm_time(mat_Tij):
    """
    mat_Tij: (T, nrow, ncol)
    Returns per-column infinity norm aggregated across time: (ncol,)
    """
    # column-wise max over rows, then max over time
    return jnp.max(jnp.max(jnp.abs(mat_Tij), axis=-2), axis=0)

def _row_inf_norm_time(mat_Tij):
    """
    mat_Tij: (T, nrow, ncol)
    Returns per-row infinity norm aggregated across time: (nrow,)
    """
    # row-wise max over cols, then max over time
    return jnp.max(jnp.max(jnp.abs(mat_Tij), axis=-1), axis=0)

def ruiz_equilibrate_mpc(
    Q, R_stage, M_stage,
    A, B,
    C, D,
    iters=5,
    eps=1e-12,
    clip=(1e-3, 1e3),
    apply_cost_scaling=True,
):
    """
    OSQP-inspired *modified Ruiz* style scaling adapted to structured MPC with variables (x,u)
    and inequality map z = Cx + Du.

    Key improvements vs your original:
      - Include dynamics (A,B) in variable scaling updates (this is typically crucial for MPC).
      - Add OSQP-like scalar objective scaling gamma (optional), applied after Ruiz iterations.

    Produces diagonal variable scaling:
      x = Sx x~, u = Su u~
    and inequality-row scaling:
      z~ = E z, f~ = E f

    Inputs:
      Q: (T+1,nx,nx)
      R_stage: (T,nu,nu)     (exclude terminal)
      M_stage: (T,nx,nu)     (exclude terminal)
      A: (T,nx,nx)
      B: (T,nx,nu)
      C: (T+1,m,nx)
      D: (T+1,m,nu)
    """
    nx = Q.shape[-1]
    nu = R_stage.shape[-1]
    m  = C.shape[1]

    dtype = Q.dtype
    Sx = jnp.ones((nx,), dtype=dtype)
    Su = jnp.ones((nu,), dtype=dtype)
    E  = jnp.ones((m,),  dtype=dtype)

    def step_fn(state, _):
        Sx, Su, E = state

        # Scale approximants under x = Sx x~, u = Su u~
        # Costs:
        Qs = (Q * Sx[None, None, :]) * Sx[None, :, None]
        Rs = (R_stage * Su[None, None, :]) * Su[None, :, None]
        Ms = (M_stage * Su[None, None, :]) * Sx[None, :, None]

        # Dynamics mapping in scaled coordinates:
        # x~_{t+1} = Sx^{-1} A Sx x~_t + Sx^{-1} B Su u~_t + ...
        # For equilibration heuristics, include magnitude of these operators.
        Sx_inv = _safe_inv(Sx, eps)
        As = (A * Sx[None, None, :]) * Sx_inv[None, :, None]  # (T,nx,nx)
        Bs = (B * Su[None, None, :]) * Sx_inv[None, :, None]  # (T,nx,nu)

        # Inequality mapping (not row-scaled yet)
        Cs = (C * Sx[None, None, :])
        Ds = (D * Su[None, None, :])

        # Variable column magnitudes (aggregate across time)
        # x columns influenced by: Qs, Ms (as x rows), As (x->x), Cs
        # Use column inf norms where meaningful and row inf norms for Ms rows.
        qcols_x = _col_inf_norm_time(Qs)                  # (nx,)
        mrows_x = _row_inf_norm_time(Ms)                  # (nx,)  (rows correspond to x components)
        acols_x = _col_inf_norm_time(As)                  # (nx,)
        ccols_x = _col_inf_norm_time(Cs)                  # (nx,)
        x_mag = jnp.maximum(jnp.maximum(qcols_x, mrows_x), jnp.maximum(acols_x, ccols_x))

        # u columns influenced by: Rs, Ms (as u cols), Bs (u->x), Ds
        rcols_u = _col_inf_norm_time(Rs)                  # (nu,)
        mcols_u = _col_inf_norm_time(Ms)                  # (nu,)
        bcols_u = _col_inf_norm_time(Bs)                  # (nu,)
        dcols_u = _col_inf_norm_time(Ds)                  # (nu,)
        u_mag = jnp.maximum(jnp.maximum(rcols_u, mcols_u), jnp.maximum(bcols_u, dcols_u))

        dSx = jnp.clip(_safe_sqrt_inv(x_mag, eps), clip[0], clip[1])
        dSu = jnp.clip(_safe_sqrt_inv(u_mag, eps), clip[0], clip[1])

        Sx_new = Sx * dSx
        Su_new = Su * dSu

        # Constraint row scaling E based on updated variable scalings
        Cs2 = (C * Sx_new[None, None, :])
        Ds2 = (D * Su_new[None, None, :])

        # row-wise max over x/u columns, then max over time
        cn = jnp.max(jnp.abs(Cs2), axis=-1)  # (T+1,m)
        dn = jnp.max(jnp.abs(Ds2), axis=-1)  # (T+1,m)
        row_mag = jnp.max(jnp.maximum(cn, dn), axis=0)     # (m,)

        dE = jnp.clip(_safe_sqrt_inv(row_mag, eps), clip[0], clip[1])
        E_new = E * dE

        return (Sx_new, Su_new, E_new), None

    (Sx, Su, E), _ = lax.scan(step_fn, (Sx, Su, E), xs=None, length=iters)

    # OSQP-like objective scaling gamma:
    # After equilibration, scale the entire objective so its typical magnitude is O(1).
    # OSQP uses a scalar based on mean ||P||_inf and max ||q||_inf; we approximate with
    # max over blocks for robustness in MPC.
    if apply_cost_scaling:
        Qs_tmp = (Q * Sx[None, None, :]) * Sx[None, :, None]
        Rs_tmp = (R_stage * Su[None, None, :]) * Su[None, :, None]
        Ms_tmp = (M_stage * Su[None, None, :]) * Sx[None, :, None]
        # A conservative scalar using infinity norms of objective blocks
        P_inf = jnp.maximum(jnp.max(jnp.abs(Qs_tmp)),
                            jnp.maximum(jnp.max(jnp.abs(Rs_tmp)), jnp.max(jnp.abs(Ms_tmp))))
        gamma = 1.0 / jnp.maximum(1.0, P_inf)
        gamma = gamma.astype(dtype)
    else:
        gamma = jnp.array(1.0, dtype=dtype)

    return RuizScaling(
        Sx=Sx,
        Su=Su,
        E=E,
        Sx_inv=_safe_inv(Sx, eps),
        Su_inv=_safe_inv(Su, eps),
        E_inv=_safe_inv(E, eps),
        gamma=gamma,
    )

def apply_ruiz_scaling_to_problem(Q, q, R, r, M, A, B, c, C, D, f, sc: RuizScaling):
    """
    Applies:
      x = Sx x~, u = Su u~,  z~ = E z,  f~ = E f
    and objective scaling:
      objective <- gamma * objective  (i.e., scale Q,R,M,q,r by gamma)

    Returns scaled (Q~,q~,R~,r~,M~,A~,B~,c~,C~,D~,f~).
    """
    Sx, Su, E = sc.Sx, sc.Su, sc.E
    Sx_inv, Su_inv = sc.Sx_inv, sc.Su_inv
    gamma = sc.gamma

    # Cost scaling under x=Sx x~, u=Su u~
    Qs = (Q * Sx[None, None, :]) * Sx[None, :, None]
    qs = q * Sx[None, :]

    Rs = (R * Su[None, None, :]) * Su[None, :, None]
    rs = r * Su[None, :]

    Ms = (M * Su[None, None, :]) * Sx[None, :, None]

    # Apply scalar objective scaling gamma (OSQP-like)
    Qs = gamma * Qs
    qs = gamma * qs
    Rs = gamma * Rs
    rs = gamma * rs
    Ms = gamma * Ms

    # Dynamics: x~_{t+1} = Sx^{-1} A Sx x~_t + Sx^{-1} B Su u~_t + Sx^{-1} c
    As = (A * Sx[None, None, :]) * Sx_inv[None, :, None]
    Bs = (B * Su[None, None, :]) * Sx_inv[None, :, None]
    cs = c * Sx_inv[None, :]

    # Inequalities: z = C x + D u, scale rows: z~ = E z
    Cs = (C * Sx[None, None, :]) * E[None, :, None]
    Ds = (D * Su[None, None, :]) * E[None, :, None]
    fs = f * E[None, :]

    return Qs, qs, Rs, rs, Ms, As, Bs, cs, Cs, Ds, fs

def scale_wy(w, y, f, sc: RuizScaling):
    # w~, y~, f~ in scaled constraint units
    return w * sc.E[None, :], y * sc.E[None, :], f * sc.E[None, :]

def unscale_primal_dual(x_tilde, u_tilde, w_tilde, y_tilde, rho, sc: RuizScaling):
    """
    Map back:
      x = Sx x~, u = Su u~
      w = E^{-1} w~, y = E^{-1} y~
      mu (Lagrange multiplier for original z<=f) = E * (rho * y~)
    """
    x = x_tilde * sc.Sx[None, :]
    u = u_tilde * sc.Su[None, :]
    w = w_tilde * sc.E_inv[None, :]
    y = y_tilde * sc.E_inv[None, :]
    mu = (rho * y_tilde) * sc.E[None, :]
    return x, u, w, y, mu

def unscale_costate(v_tilde, sc: RuizScaling):
    """
    If v is interpreted as gradient/costate in original x-units:
      - variable change x = Sx x~ => gradients transform as v = Sx^{-T} v~ = Sx^{-1} v~ (diag)
      - objective scaling objective <- gamma * objective => gradients scale by gamma,
        so to recover original gradients divide by gamma.
    """
    return (v_tilde * sc.Sx_inv[None, :]) / sc.gamma

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
    m = mask.astype(vals.dtype)[:, None, None]
    return cache.at[level].set(vals * m)

def right_solve(M, B, reg=1e-6):
    """
    Compute X = B @ inv(M) without forming inv(M).
    Uses (M^T) X^T = B^T => X = solve(M^T, B^T)^T.
    M: (T,n,n), B: (T,n,n) -> X: (T,n,n)
    """
    n = M.shape[-1]
    I = jnp.eye(n, dtype=M.dtype)[None, :, :]
    Mreg = M + reg * I
    Xt = jnp.linalg.solve(jnp.swapaxes(Mreg, -1, -2), jnp.swapaxes(B, -1, -2))
    return jnp.swapaxes(Xt, -1, -2)

def _combine_acp_all(next_block, prev_block, n):
    """
    next_block, prev_block: (T, 3n, n)
    Returns:
      combined: (T, 3n, n)
      Ar, Al, ArC, AlP, C_new, P_new: (T, n, n)
    """
    dtype = prev_block.dtype
    I = jnp.eye(n, dtype=dtype)[None, :, :]

    A_l = prev_block[:, 0:n, :]
    C_l = prev_block[:, n:2*n, :]
    P_l = prev_block[:, 2*n:3*n, :]

    A_r = next_block[:, 0:n, :]
    C_r = next_block[:, n:2*n, :]
    P_r = next_block[:, 2*n:3*n, :]

    # inv1 = jnp.linalg.inv(I + C_l @ P_r)
    # inv2 = jnp.linalg.inv(I + P_r @ C_l)

    # Ar = A_r @ inv1
    # Al = jnp.swapaxes(A_l, -1, -2) @ inv2
    M1 = I + C_l @ P_r
    M2 = I + P_r @ C_l

    Ar = right_solve(M1, A_r, reg=1e-4)                              # Ar = A_r @ inv(M1)
    Al = right_solve(M2, jnp.swapaxes(A_l, -1, -2), reg=1e-4)   

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

    # Downsweep
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
    """Dual LQR solve: v[t] = P[t] X[t] + p[t]."""
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
    CtC = jnp.einsum('tmi,tmj->tij', C, C)
    DtD = jnp.einsum('tmi,tmj->tij', D, D)
    CtD = jnp.einsum('tmi,tmj->tij', C, D)

    Ct_s = jnp.einsum('tmi,tm->ti', C, s_bar)
    Dt_s = jnp.einsum('tmi,tm->ti', D, s_bar)

    tilde_Q = Q + rho * CtC
    tilde_q = q - rho * Ct_s

    tilde_R = R + rho * DtD
    tilde_r = r - rho * Dt_s

    tilde_M = M + rho * CtD
    return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M

def admm_residuals(z, w, w_prev, y, rho, eps_abs=1e-2, eps_rel=1e-2):
    """
    Infinity norm residuals and thresholds.
    """
    r = z - w
    s = rho * (w - w_prev)

    r_norm = jnp.linalg.norm(r.reshape(-1), ord=jnp.inf)
    s_norm = jnp.linalg.norm(s.reshape(-1), ord=jnp.inf)

    z_norm = jnp.linalg.norm(z.reshape(-1), ord=jnp.inf)
    w_norm = jnp.linalg.norm(w.reshape(-1), ord=jnp.inf)
    y_norm = jnp.linalg.norm(y.reshape(-1), ord=jnp.inf)

    eps_pri = eps_abs + eps_rel * jnp.maximum(z_norm, w_norm)
    eps_dual = eps_abs + eps_rel * (rho * y_norm)

    return r_norm, s_norm, eps_pri, eps_dual

def adaptive_rho_update(rp_norm, rd_norm, rho,
                        clip_min=0.2, clip_max=5,
                        rho_min=1e-2, rho_max=1e5,
                        eps=1e-12):
    scale = rp_norm / (rd_norm + eps)
    scale = jnp.clip(scale, clip_min, clip_max)
    rho_new = jnp.clip(rho * scale, rho_min, rho_max)
    updated = rho_new != rho
    return rho_new, updated

def rho_update_y(rp_norm, rd_norm, rho, y, rho_max):
    rho_new, updated = adaptive_rho_update(rp_norm, rd_norm, rho, rho_max=rho_max)
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
        Rt = 0.5 * (tilde_R[t] + tilde_R[t].T)
        Rt = Rt + 1e-8 * jnp.eye(Rt.shape[0], dtype=Rt.dtype)
        I = jnp.eye(tilde_R[t].shape[0], dtype=Rt.dtype)
        return jnp.linalg.solve(Rt, I)

    Rinv = vmap(chol_inv)(jnp.arange(T))
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))
    MRinv = vmap(lambda t: tilde_M[t] @ Rinv[t])(jnp.arange(T))

    elems = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    A - vmap(lambda t: BRinv[t] @ tilde_M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n], dtype=tilde_Q.dtype),
                ]
            ),
            jnp.concatenate(
                [
                    vmap(lambda t: BRinv[t] @ B[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n], dtype=tilde_Q.dtype),
                ]
            ),
            tilde_Q
            - jnp.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ tilde_M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n], dtype=tilde_Q.dtype),
                ]
            ),
        ],
        axis=1,
    )

    return elems, BRinv, MRinv

def generate_leaf_bp(c, BRinv, MRinv, tilde_r, tilde_q, T, n):
    c_stage = c - vmap(lambda t: BRinv[t] @ tilde_r[t])(jnp.arange(T))
    c0 = jnp.concatenate([c_stage, jnp.zeros((1, n), dtype=c.dtype)], axis=0)

    p_stage = vmap(lambda t: MRinv[t] @ tilde_r[t])(jnp.arange(T))
    p0 = tilde_q - jnp.concatenate([p_stage, jnp.zeros((1, n), dtype=tilde_q.dtype)], axis=0)

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
    T = B.shape[0]
    def one(t):
        BtP = B[t].T @ P[t + 1]
        H  = BtP @ A[t] + tilde_M[t].T
        G  = tilde_R[t] + BtP @ B[t]
        return scipy.linalg.solve(G, -H)
    return vmap(one)(jnp.arange(T))

def constrained_solve(cfg: ADMMConfig,
                      Q, q, R, r, M,
                      A, B, c,
                      C, D, f,
                      w, y, rho):
    """
    ADMM loop on inequality: z = Cx + Du, w = proj(z+y) onto (-inf, f], scaled form.

    This version incorporates:
      - Ruiz-style variable + inequality-row scaling with dynamics-aware updates
      - OSQP-like scalar objective scaling gamma
    Solves entirely in (~) coordinates, returning unscaled outputs.
    """
    rho_max = cfg.rho_max

    # Shapes
    T = A.shape[0]
    nx = Q.shape[-1]
    nu = R.shape[-1]
    m  = C.shape[1]
    Tp1 = T + 1

    # Pad terminal for consistency (your original approach)
    R_pad = jnp.concatenate([R, jnp.zeros((1, nu, nu), dtype=R.dtype)], axis=0)
    r_pad = jnp.concatenate([r, jnp.zeros((1, nu), dtype=r.dtype)], axis=0)
    M_pad = jnp.concatenate([M, jnp.zeros((1, nx, nu), dtype=M.dtype)], axis=0)

    # ---- Modified Ruiz (dynamics-aware) computed from base (unaugmented) blocks ----
    sc = ruiz_equilibrate_mpc(Q, R, M, A, B, C, D, iters=5, apply_cost_scaling=True, eps=1e-6)

    # Apply scaling to whole problem
    Qs, qs, Rs, rs, Ms, As, Bs, cs, Cs, Ds, fs = apply_ruiz_scaling_to_problem(
        Q, q, R_pad, r_pad, M_pad, A, B, c, C, D, f, sc
    )

    # Scale warm starts (constraint space)
    w_s, y_s, f_s = scale_wy(w, y, f, sc)

    # Init in scaled coordinates
    init_x = jnp.zeros((Tp1, nx), dtype=Q.dtype)   # x~
    init_u = jnp.zeros((Tp1, nu), dtype=Q.dtype)   # u~
    init_w = w_s
    init_y = y_s
    rho0 = rho
    p_init = jnp.zeros((Tp1, nx), dtype=Q.dtype)

    # Initial augmented LQR precompute
    tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M = admm_augment_xu(
        Qs, qs, Rs, rs, Ms, Cs, Ds, init_w, init_y, rho0
    )
    elems_acp, BRinv, MRinv = generate_leaf(tilde_Q, tilde_R, tilde_M, As, Bs)
    out_acp, cache = associative_scan_cache_acp_jax(elems_acp, Tp1, nx, reverse=True)
    P = out_acp[:, -nx:, :]
    K = get_K(tilde_R, tilde_M, As, Bs, P)

    def one_iter(carry):
        (it, tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M,
         x_bar, u_bar, y_bar, w_prev, rho, cache, BRinv, MRinv, P, p_prev, K,
         rp_norm, rd_norm, eps_pri, eps_dual, converged) = carry

        # -------- Solve unconstrained LQR subproblem (scaled) --------
        c0, p0 = generate_leaf_bp(cs[1:], BRinv, MRinv, tilde_r, tilde_q, T, nx)
        b, p = associative_scan_use_cache_cp_jax(c0, p0, Tp1, cache, reverse=True)

        k = get_k(tilde_R, tilde_r, Bs, P, p, cs[1:])

        x_bar, u_stage = rollout_gpu(K, k, cs[0], As, Bs, cs[1:])
        u_bar = jnp.pad(u_stage, ((0, 1), (0, 0)))  # (T+1, nu) in scaled u~

        # z~ = Cs x~ + Ds u~
        z_bar = (
            jnp.einsum('tmi,ti->tm', Cs, x_bar) +
            jnp.einsum('tmi,ti->tm', Ds, u_bar)
        )

        # -------- Project onto constraint set (scaled) --------
        w_new = jnp.minimum(z_bar + y_bar, f_s)

        # -------- Dual update (scaled form) --------
        y_new = y_bar + (z_bar - w_new)

        # -------- Residuals / stopping --------
        rp_norm, rd_norm, eps_pri, eps_dual = admm_residuals(
            z_bar, w_new, w_prev, y_new, rho,
            eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel
        )
        converged = jnp.logical_and(rp_norm <= eps_pri, rd_norm <= eps_dual)

        # Adaptive rho (gated)
        do_rho_update = (it % cfg.rho_update_frequency) == 0

        def update_fn(_):
            rho_upd, y_upd, updated = rho_update_y(
                rp_norm, rd_norm,
                rho, y_new, rho_max
            )
            return rho_upd, y_upd, updated

        def no_update_fn(_):
            return rho, y_new, jnp.array(False)

        rho_new, y_upd, rho_updated = lax.cond(do_rho_update, update_fn, no_update_fn, operand=None)

        def cache_update(_):
            tilde_Q2, tilde_q2, tilde_R2, tilde_r2, tilde_M2 = admm_augment_xu(
                Qs, qs, Rs, rs, Ms, Cs, Ds, w_new, y_upd, rho_new
            )
            elems_acp2, BRinv2, MRinv2 = generate_leaf(tilde_Q2, tilde_R2, tilde_M2, As, Bs)
            out_acp2, cache2 = associative_scan_cache_acp_jax(elems_acp2, Tp1, nx, reverse=True)
            P2 = out_acp2[:, -nx:, :]
            K2 = get_K(tilde_R2, tilde_M2, As, Bs, P2)
            return tilde_Q2, tilde_q2, tilde_R2, tilde_r2, tilde_M2, cache2, BRinv2, MRinv2, P2, K2

        def no_cache_update(_):
            return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache, BRinv, MRinv, P, K

        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache, BRinv, MRinv, P, K = lax.cond(
            rho_updated, cache_update, no_cache_update, operand=None
        )

        return (it + 1, tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M,
                x_bar, u_bar, y_upd, w_new, rho_new,
                cache, BRinv, MRinv, P, p, K,
                rp_norm, rd_norm, eps_pri, eps_dual, converged)

    def cond_fun(carry):
        it = carry[0]
        converged = carry[-1]
        return jnp.logical_and(it < cfg.max_iterations, jnp.logical_not(converged))

    init = (
        jnp.array(1, dtype=jnp.int32),
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M,
        init_x, init_u, init_y, init_w,
        jnp.array(rho0, dtype=Q.dtype),
        cache, BRinv, MRinv, P, p_init, K,
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(False)
    )

    out = jax.lax.while_loop(cond_fun, one_iter, init)

    (it, _, _, _, _, _, x_bar, u_bar, y_bar, w_bar, rho_final,
     _, _, _, P_final, p_final, _, rp_norm, rd_norm, eps_pri, eps_dual, converged) = out

    # Costate (scaled), then unscale to original x-units (and undo objective scaling gamma)
    v_tilde = dual_lqr(x_bar, P_final, p_final)
    v_un = unscale_costate(v_tilde, sc)

    # Unscale primal/dual to original units
    x_un, u_un, w_un, y_un, mu_un = unscale_primal_dual(
        x_bar, u_bar[:-1], w_bar, y_bar, rho_final, sc
    )

    jax.debug.print(
        "ADMM done: Total Iterations={} converged={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e}) Rho0 {:.3e} gamma {:.3e}",
        it - 1, converged, rho_final, rp_norm, eps_pri, rd_norm, eps_dual, rho0, sc.gamma
    )

    return x_un, u_un, v_un, w_un, y_un, rho_final, mu_un, converged
