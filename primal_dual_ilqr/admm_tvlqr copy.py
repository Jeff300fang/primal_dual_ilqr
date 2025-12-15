from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import jit, lax, scipy, vmap
@dataclass
class ADMMConfig:
    rho_update_frequency: int = 25
    max_iterations: int = 400
    eps_abs: float = 1e-2
    eps_rel: float = 1e-2
    condense_block_size: int = 1

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

@jit
def tvlqr_gpu(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    This is a O(log T) parallel time complexity implementation, based on
    https://ieeexplore.ieee.org/document/9697418.

    Args:
      Q: [T+1, n, n] numpy array.
      q: [T+1, n]    numpy array.
      R: [T, m, m]   numpy array.
      r: [T, m]      numpy array.
      M: [T, n, m]   numpy array.
      A: [T, n, n]   numpy array.
      B: [T, n, m]   numpy array.
      c: [T, n]      numpy array.
      delta: Enforces positive definiteness by ensuring smallest eigenval > delta.

    Returns:
      K: [T, m, n] Gains
      k: [T, m] Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n] numpy array encoding initial value function.
      p: [T+1, n] numpy array encoding initial value function.
    """
    T = Q.shape[0] - 1
    n = Q.shape[1]

    def fn(next, prev):
        def decompose(elem):
            return (
                elem[:n],
                elem[n],
                elem[n + 1 : 2 * n + 1],
                elem[2 * n + 1],
                elem[-n:],
            )

        A_l, c_l, C_l, p_l, P_l = decompose(prev)
        A_r, c_r, C_r, p_r, P_r = decompose(next)

        ArIClPr_inv = A_r @ jnp.linalg.inv(jnp.eye(n) + C_l @ P_r)
        AlTIPrCl_inv = A_l.T @ jnp.linalg.inv(jnp.eye(n) + P_r @ C_l)

        A_new = ArIClPr_inv @ A_l
        c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
        C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
        p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
        P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

        return jnp.concatenate(
            [
                A_new,
                c_new.reshape(1, n),
                C_new,
                p_new.reshape(1, n),
                P_new,
            ]
        )

    def chol_inv(t):
        f = scipy.linalg.cho_factor(R[t])
        m = R[t].shape[0]
        return scipy.linalg.cho_solve(f, jnp.eye(m))

    Rinv = vmap(chol_inv)(jnp.arange(T))
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))
    MRinv = vmap(lambda t: M[t] @ Rinv[t])(jnp.arange(T))

    elems = jnp.concatenate(
        [
            # The A matrices.
            jnp.concatenate(
                [
                    A - vmap(lambda t: BRinv[t] @ M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            # The c vectors (b, in the notation of https://ieeexplore.ieee.org/document/9697418).
            jnp.concatenate(
                [
                    (c - vmap(lambda t: BRinv[t] @ r[t])(jnp.arange(T))).reshape(
                        [T, 1, n]
                    ),
                    jnp.zeros([1, 1, n]),
                ]
            ),
            # The C matrices.
            jnp.concatenate(
                [
                    vmap(lambda t: BRinv[t] @ B[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            # The p vectors (-eta, in the notation of https://ieeexplore.ieee.org/document/9697418).
            q.reshape([T + 1, 1, n])
            - jnp.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ r[t])(jnp.arange(T)).reshape([T, 1, n]),
                    jnp.zeros([1, 1, n]),
                ]
            ),
            # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
            Q
            - jnp.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
        ],
        axis=1,
    )

    result = lax.associative_scan(lambda r, l: vmap(fn)(r, l), elems, reverse=True)

    P = result[:, -n:, :]
    p = result[:, 2 * n + 1, :]

    def getKs(t):
        # symmetrize = lambda x: 0.5 * (x + x.T)

        BtP = B[t].T @ P[t + 1]
        BtPA = BtP @ A[t]

        H = BtPA + M[t].T
        h = B[t].T @ p[t + 1] + BtP @ c[t] + r[t]

        # G = symmetrize(R[t] + BtP @ B[t])

        # f = scipy.linalg.cho_factor(G)
        K_k = scipy.linalg.solve(R[t] + BtP @ B[t], -jnp.hstack((H, h.reshape([-1, 1]))))
        K = K_k[:, :-1]
        k = K_k[:, -1]

        return K, k

    K, k = vmap(getKs)(jnp.arange(T))

    return K, k, P, p


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

def constrained_solve(cfg: ADMMConfig, Q, q, R, r, M, A, B, c, C, D, f):
    rho0 = 0.1
    T = A.shape[0]
    m = C.shape[1]
    nx = Q.shape[-1]
    nu = R.shape[-1]

    # Pad terminal for consistency with x_{T} term
    R = jnp.concatenate([R, jnp.zeros((1, nu, nu), dtype=R.dtype)], axis=0)
    r = jnp.concatenate([r, jnp.zeros((1, nu), dtype=r.dtype)], axis=0)
    M = jnp.concatenate([M, jnp.zeros((1, nx, nu), dtype=M.dtype)], axis=0)

    # --- optional: adaptive rho update (Boyd-style), JAX friendly ---
    def adaptive_rho_update(rp_norm, rd_norm, rho,
                            mu=10.0, tau_inc=2.0, tau_dec=2.0,
                            rho_min=1e-2, rho_max=1e3):
        inc = rp_norm > mu * rd_norm
        dec = rd_norm > mu * rp_norm

        rho_inc = jnp.minimum(rho * tau_inc, rho_max)
        rho_dec = jnp.maximum(rho / tau_dec, rho_min)

        rho_new = jnp.where(inc, rho_inc,
                   jnp.where(dec, rho_dec, rho))
        updated = rho_new != rho
        return rho_new, updated

    # --- one ADMM iteration ---
    def one_iter(carry):
        (it, x_bar, u_bar, w_bar, y_bar, v, w_prev, rho,
         rp_norm, rd_norm, eps_pri, eps_dual, converged) = carry

        # Augment LQR quadratic
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M = admm_augment_xu(
            Q, q, R, r, M, C, D, w_bar, y_bar, rho
        )

        # Solve unconstrained LQR subproblem
        K, k, P, p = tvlqr_gpu(tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, A, B, c[1:])
        x_bar, u_stage = rollout_gpu(K, k, c[0], A, B, c[1:])
        u_bar = jnp.pad(u_stage, ((0, 1), (0, 0)))  # (T+1, nu)
        v = dual_lqr(x_bar, P, p)

        # z = Cx + Du
        z_bar = (
            jnp.einsum('tmi,ti->tm', C, x_bar) +
            jnp.einsum('tmi,ti->tm', D, u_bar)
        )

        # Project onto constraint set (upper bounds only)
        w_new = jnp.minimum(z_bar + y_bar, f)

        # Dual update (scaled form)
        y_new = y_bar + (z_bar - w_new)

        # Residual norms and tolerances
        rp_norm, rd_norm, eps_pri, eps_dual = admm_residuals(
            z_bar, w_new, w_prev, y_new, rho,
            eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel
        )

        # Convergence check
        converged = jnp.logical_and(rp_norm <= eps_pri, rd_norm <= eps_dual)

        # Adaptive rho (gated)
        do_rho_update = (it % cfg.rho_update_frequency) == 0
        rho_candidate, rho_updated = adaptive_rho_update(rp_norm, rd_norm, rho)
        rho_new = jnp.where(do_rho_update, rho_candidate, rho)

        # Optional (recommended): rescale y when rho changes significantly
        # y := (rho_old / rho_new) * y  (scaled ADMM consistency)
        y_new = jnp.where(rho_new != rho, (rho / rho_new) * y_new, y_new)

        # jax.debug.print(
        #     "ADMM: it={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e})",
        #     it, rho, rp_norm, eps_pri, rd_norm, eps_dual
        # )

        return (it + 1, x_bar, u_bar, w_new, y_new, v, w_new, rho_new,
                rp_norm, rd_norm, eps_pri, eps_dual, converged)

    # --- loop condition: keep going until max_iters OR converged ---
    def cond_fun(carry):
        it = carry[0]
        converged = carry[-1]
        return jnp.logical_and(it < cfg.max_iterations, jnp.logical_not(converged))

    # Init
    init_x = jnp.zeros((T + 1, nx), dtype=Q.dtype)
    init_u = jnp.zeros((T + 1, nu), dtype=Q.dtype)
    init_w = jnp.zeros((T + 1, m), dtype=Q.dtype)
    init_y = jnp.zeros((T + 1, m), dtype=Q.dtype)
    init_v = jnp.zeros((T + 1, nx), dtype=Q.dtype)

    init = (
        jnp.array(0, dtype=jnp.int32),  # it
        init_x, init_u, init_w, init_y, init_v,
        init_w,                          # w_prev
        jnp.array(rho0, dtype=Q.dtype),  # rho
        jnp.array(jnp.inf, dtype=Q.dtype),  # rp_norm
        jnp.array(jnp.inf, dtype=Q.dtype),  # rd_norm
        jnp.array(jnp.inf, dtype=Q.dtype),  # eps_pri
        jnp.array(jnp.inf, dtype=Q.dtype),  # eps_dual
        jnp.array(False)                    # converged
    )

    out = jax.lax.while_loop(cond_fun, one_iter, init)

    it, x_bar, u_bar, w_bar, y_bar, v, _, rho, rp_norm, rd_norm, eps_pri, eps_dual, converged = out

    # If you want logging, print once at the end (safe in jit)
    jax.debug.print(
        "ADMM done: it={} converged={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e})",
        it, converged, rho, rp_norm, eps_pri, rd_norm, eps_dual
    )

    return x_bar, u_bar[:-1], v

