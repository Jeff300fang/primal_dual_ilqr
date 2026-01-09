from __future__ import annotations
from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
from jax import lax, vmap
from mpx.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import tvlqr_gpu
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import constrained_solve
from dataclasses import dataclass

@dataclass(frozen=True)
class SLSConfig:
    max_sls_iterations: int = 2
    sls_primal_tol: float = 1e-2
    enable_fastsls: bool = True


@jax.jit
def calculate_cost(Q_bar, R_bar, C, D, eta):
    eta = jnp.asarray(eta).reshape(-1)
    eta = jnp.maximum(eta, 0.0)

    s = jnp.sqrt(eta)
    Cs = C[:-1] * s[:, None]
    Ds = D[:-1] * s[:, None]

    Cx  = Cs.T @ Cs + Q_bar
    Cxu = Cs.T @ Ds
    Cu  = Ds.T @ Ds + R_bar

    return Cx, Cxu, Cu

@jax.jit
def calculate_phis(A, B, Cx, Cxu, Cu, E):
    """
    Same as your current associative-scan version, but robust to accidentally
    passing A/B with length T+1. Ensures Phi_u has leading dimension T (stage length).
    """
    # Define stage horizon from Cu (or B) which should be length T
    T = Cu.shape[0]                 # stage length
    nx = A.shape[1]
    nu = B.shape[-1]
    Tp1 = T + 1
    nw = E.shape[-1]

    # Slice A, B to stage horizon in case they were passed as length T+1
    A = A[:T]
    B = B[:T]

    zeros_q = jnp.zeros((Tp1, nx), dtype=A.dtype)
    zeros_r = jnp.zeros((T,  nu), dtype=A.dtype)
    zeros_c = jnp.zeros((T,  nx), dtype=A.dtype)

    # ---- 1) Solve K_{k,j} for j=0..T-1, pad j=T with zeros ----
    def solve_one_j(j):
        Qj = Cx[:, j, :, :]     # [T+1, nx, nx]
        Rj = Cu[:, j, :, :]     # [T,   nu, nu]
        Mj = Cxu[:, j, :, :]    # [T,   nx, nu]
        K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
        return K                # [T, nu, nx]

    K_all = jax.vmap(solve_one_j)(jnp.arange(T))      # [T(j), T(k), nu, nx]
    K_kj_core = jnp.swapaxes(K_all, 0, 1)             # [T(k), T(j), nu, nx]
    K_lastcol = jnp.zeros((T, 1, nu, nx), dtype=A.dtype)
    K_kj = jnp.concatenate([K_kj_core, K_lastcol], axis=1)  # [T, T+1, nu, nx]

    # ---- 2) Closed-loop transitions F[k,j] ----
    BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)        # [T, T+1, nx, nx]
    F  = A[:, None, :, :] + BK                        # [T, T+1, nx, nx]

    # Force j=T column to identity (no propagation for the j=T column)
    I = jnp.eye(nx, dtype=A.dtype)
    F = F.at[:, T].set(I)

    # ---- 3) associative_scan prefix products ----
    t_idx = jnp.arange(T)[:, None]       # [T,1]
    j_idx = jnp.arange(Tp1)[None, :]     # [1,T+1]
    use_F = (t_idx >= j_idx)             # [T, T+1]
    elems = jnp.where(use_F[:, :, None, None], F, I)  # [T, T+1, nx, nx]

    def compose(l, r):
        return jnp.einsum("...ab,...bc->...ac", r, l)

    P = lax.associative_scan(compose, elems, axis=0)  # [T, T+1, nx, nx]

    # ---- 4) Build Phi_x ----
    Phix_1toT = jnp.einsum("tjab,jbn->tjan", P, E)     # [T, T+1, nx, nw]
    Phi_x = jnp.concatenate(
        [jnp.zeros((1, Tp1, nx, nw), dtype=A.dtype), Phix_1toT],
        axis=0
    )  # [T+1, T+1, nx, nw]

    Phi_x = Phi_x.at[jnp.arange(Tp1), jnp.arange(Tp1)].set(E)

    k_idx_full = jnp.arange(Tp1)[:, None]             # [T+1,1]
    valid_x = (k_idx_full >= j_idx)                   # [T+1, T+1]
    Phi_x = Phi_x * valid_x[:, :, None, None]

    # ---- 5) Phi_u ----
    Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])    # [T, T+1, nu, nw]
    k_idx = jnp.arange(T)[:, None]                             # [T,1]
    valid_u = (k_idx >= j_idx)                                 # [T, T+1]
    Phi_u = Phi_u * valid_u[:, :, None, None]

    return Phi_x, Phi_u

@jax.jit
def get_controller(Q, R, A, B, C, D, E, eta_stage, eta_f):
    """
    eta_stage: [T, T, nc]
    eta_f:     [T+1, nc]   (j = 0..T)
    """
    T, nx, _ = A.shape
    nc = C.shape[1]

    js = jnp.arange(T)
    ks = jnp.arange(T)

    def blocks_for_k(k):
        def blocks_for_j(j):
            return calculate_cost(Q[k], R[k], C[k], D[k], eta_stage[k, j])
        return vmap(blocks_for_j)(js)

    Cx_kj, Cxu_kj, Cu_kj = vmap(blocks_for_k)(ks)  # [T,T,...]

    Cterm = C[-1, :-1]  # [nc, nx]
    def terminal_Cx_for_j(j):
        w = eta_f[j]  # <-- correct indexing by j
        return (Cterm.T * w[None, :]) @ Cterm + Q[T]

    Cx_Nj = vmap(terminal_Cx_for_j)(jnp.arange(T))  # [T, nx, nx]
    Cx = jnp.concatenate([Cx_kj, Cx_Nj[None, ...]], axis=0)      # [T+1, T, nx, nx]

    Phi_x, Phi_u = calculate_phis(A, B, Cx, Cxu_kj, Cu_kj, E)
    return Phi_x, Phi_u

@jax.jit
def get_betas(C, D, Phi_x, Phi_u):
    """
    Shape-consistent with fixed phis.

    C:     [T+1, nc, nx]
    D:     [T,   nc, nu]
    Phi_x: [T+1, T+1, nx, nw]
    Phi_u: [T,   T+1, nu, nw]

    Returns:
      beta: [T+1, T+1, nc]  (we’ll fill stage rows k=0..T-1, terminal row k=T)
            Downstream can still slice as needed.
    """
    T = Phi_u.shape[0]
    Tp1 = T + 1
    nc = C.shape[1] - 1

    # Stage: k=0..T-1
    # term_x[k,j,i,:] = C[k,i] @ Phi_x[k,j]
    term_x = jnp.einsum("kix,kjxn->kjin", C[:-1, :-1], Phi_x[:-1])  # [T, T+1, nc, nw]
    # term_u[k,j,i,:] = D[k,i] @ Phi_u[k,j]
    term_u = jnp.einsum("kiu,kjun->kjin", D[:-1, :-1], Phi_u)            # [T, T+1, nc, nw]
    gPhi = term_x + term_u
    beta_stage = jnp.sum(gPhi * gPhi, axis=-1)                 # [T, T+1, nc]

    # Mask invalid entries (j > k) to 0 to keep your triangular structure
    k_idx = jnp.arange(T)[:, None]          # [T,1]
    j_idx = jnp.arange(Tp1)[None, :]        # [1,T+1]
    mask = (j_idx <= k_idx)                 # [T, T+1]
    beta_stage = beta_stage * mask[:, :, None]

    # Terminal row (k=T): only state constraints contribute (no D at terminal)
    # beta_term[j,i] = || C[T,i] @ Phi_x[T,j] ||^2
    gPhi_term = jnp.einsum("ix,jxn->jin", C[-1, :-1], Phi_x[-1])    # [T+1, nc, nw]
    beta_term = jnp.sum(gPhi_term * gPhi_term, axis=-1)        # [T+1, nc]

    # Pack into one tensor like your old code expects.
    beta = jnp.zeros((Tp1, Tp1, nc), dtype=Phi_x.dtype)
    beta = beta.at[:-1, :, :].set(beta_stage)                  # k=0..T-1
    beta = beta.at[-1, :, :].set(beta_term)                    # k=T

    return beta

@jax.jit
def get_constraint_tightenings(betas, eps_beta=1e-6):
    """
    betas: [T+1, T+1, nc]
    h_ct[k] = sum_{j=0}^{k-1} sqrt(betas[k,j,:]) + eps_beta
    """
    T1, T1b, nc = betas.shape
    # sanity: T1b == T1

    s = jnp.sqrt(jnp.maximum(betas, 0.0))  # [T+1, T+1, nc]

    k_idx = jnp.arange(T1)[:, None]        # [T+1,1]
    j_idx = jnp.arange(T1)[None, :]        # [1,T+1]
    valid = (j_idx <= k_idx)                # [T+1, T+1]
    s = s * valid[:, :, None]

    h_ct = jnp.sum(s, axis=1)              # [T+1, nc]
    # return h_ct
    h_ct = h_ct + eps_beta
    h_ct_xy = jnp.sqrt(
        h_ct[:, 0] ** 2 + h_ct[:, 1] ** 2
    )[:, None]                                      # [T+1, 1]

    return jnp.concatenate([h_ct, h_ct_xy], axis=1) # [T+1, nc+1]

@jax.jit
def get_etas(mus, betas, eps=1e-12):
    """
    Drop-in signature: (mus, betas, eps) -> (eta, eta_f)

    mus:   [T+1, nc]         (inequality multipliers per time, terminal at mus[T])
    betas: [T+1, T+1, nc]    (beta[T, j, :] is terminal beta_f[j])

    Returns:
      eta:   [T,   T,   nc]   for stage costs (k=0..T-1, j=0..T-1)
      eta_f: [T+1,      nc]   for terminal boundary (j=0..T)
    """
    Tp1 = mus.shape[0]
    T = Tp1 - 1

    # stage eta[k,j,:] = mu[k,:] / (2*sqrt(beta[k,j,:]))
    mu_k = mus[:-1, :-1]                    # [T, nc]
    beta_kj = betas[:-1, :-1, :]       # [T, T, nc]
    eta = (mu_k[:, None, :] /
           (2.0 * jnp.sqrt(jnp.maximum(beta_kj, eps))))

    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(T)[None, :]
    eta = eta * (k_idx >= j_idx)[:, :, None]
    eta = jnp.maximum(eta, 0.0)

    # terminal eta_f[j,:] = mu_f / (2*sqrt(beta_f[j,:]))
    mu_f = mus[-1, :-1]                     # [nc]
    beta_f = betas[-1, :, :]           # [T+1, nc]  <-- terminal row k=T
    eta_f = (mu_f[None, :] /
             (2.0 * jnp.sqrt(jnp.maximum(beta_f, eps))))
    eta_f = jnp.maximum(eta_f, 0.0)    # [T+1, nc]

    return eta, eta_f

@jax.jit
def _scaled_primal_diff(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Returns a scaled infinity-norm difference:
        ||a-b||_inf / max(1, ||b||_inf)
    """
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(1.0, jnp.max(jnp.abs(b)))
    return num / (den + eps)

@jax.jit
def primal_convergence_metric(
    X_new: jnp.ndarray, U_new: jnp.ndarray,
    X_old: jnp.ndarray, U_old: jnp.ndarray
) -> jnp.ndarray:
    """
    Single scalar convergence metric = max of scaled diffs across primal blocks.
    """
    mX = _scaled_primal_diff(X_new, X_old)
    mU = _scaled_primal_diff(U_new, U_old)
    return jnp.maximum(mX, mU)

@partial(jit, static_argnums=(0, 15))
def fast_sls_solve_gpu(cfg, Q: jnp.ndarray, q: jnp.ndarray,
                       R: jnp.ndarray, r: jnp.ndarray,
                       M: jnp.ndarray,
                       A: jnp.ndarray, B: jnp.ndarray, c: jnp.ndarray,
                       C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                       w: jnp.ndarray, y: jnp.ndarray, rho: jnp.ndarray, # ADMM Params
                       sls_config: SLSConfig, E: jnp.ndarray, Q_bar: jnp.ndarray, R_bar: jnp.ndarray):
    # Solve Nominal Trajectory
    
    Tp1 = Q.shape[0]
    nx  = Q.shape[1]
    nu  = R.shape[1]
    nc  = w.shape[1] - 1
    T   = Tp1 - 1

    beta0 = jnp.ones((Tp1, Tp1, nc), dtype=Q.dtype) * 1e-10
    x0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)
    u0 = jnp.zeros((T, nu),  dtype=Q.dtype)
    v0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)

    i0 = jnp.array(0, dtype=jnp.int32)
    converged0 = jnp.array(False)

    max_iter = jnp.array(sls_config.max_sls_iterations, dtype=jnp.int32)
    tol = jnp.array(sls_config.sls_primal_tol, dtype=Q.dtype)

    # carry = (i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, admm_converged)
    h_ct0 = get_constraint_tightenings(beta0)
    Phi_x0 = jnp.zeros((Tp1, Tp1, nx, nx))
    Phi_u0 = jnp.zeros((T, Tp1, nu, nx))
    carry0 = (i0, beta0, x0, u0, v0, w, y, rho, converged0, converged0, h_ct0, Phi_x0, Phi_u0)

    def cond_fn(carry):
        i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _, _, _, _ = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_fn(carry):
        i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _, h_ct, Phi_x, Phi_u = carry

        prev_rho = rho
        x_prev = x_curr
        u_prev = u_curr

        tightened_constraints = f - h_ct

        # (your reset logic)
        w = jnp.zeros_like(w)
        y = jnp.zeros_like(y)
        rho = jnp.array(10)

        x_curr, u_curr, v_curr, w, y, rho, mu, converged_admm = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C, D, tightened_constraints, w, y, rho
        )

        metric = primal_convergence_metric(x_curr, u_curr, x_prev, u_prev)
        converged_now = metric <= tol

        # If we are converged, skip eta/controller/beta/tightening updates.
        def do_updates(args):
            beta, h_ct, Phi_x, Phi_u, mu = args

            eta_stage, eta_f = get_etas(mu, beta)
            Phi_x_new, Phi_u_new = get_controller(Q_bar, R_bar, A, B, C, D, E, eta_stage, eta_f)
            beta_new = get_betas(C, D, Phi_x_new, Phi_u_new)
            h_ct_new = get_constraint_tightenings(beta_new)

            return beta_new, h_ct_new, Phi_x_new, Phi_u_new

        def skip_updates(args):
            beta, h_ct, Phi_x, Phi_u, mu = args
            return beta, h_ct, Phi_x, Phi_u

        beta, h_ct, Phi_x, Phi_u = lax.cond(
            converged_now,
            skip_updates,
            do_updates,
            (beta, h_ct, Phi_x, Phi_u, mu),
        )

        # (your rho/y scaling logic)
        rho = jnp.maximum(jnp.minimum(rho, 1e3) * 0.5, 0.1)
        y = prev_rho / rho * y

        converged = jnp.logical_or(converged, converged_now)

        return (
            i + jnp.array(1, dtype=jnp.int32),
            beta, x_curr, u_curr, v_curr, w, y, rho, converged, converged_admm,
            h_ct, Phi_x, Phi_u
        )

    carryN = jax.lax.while_loop(cond_fn, body_fn, carry0)

    _, betaN, xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct, Phi_x, Phi_u = carryN
    return xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct, Phi_x, Phi_u