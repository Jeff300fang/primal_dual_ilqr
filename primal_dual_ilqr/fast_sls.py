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
    Cs = C * s[:, None]
    Ds = D * s[:, None]

    Cx  = Cs.T @ Cs + Q_bar
    Cxu = Cs.T @ Ds
    Cu  = Ds.T @ Ds + R_bar

    return Cx, Cxu, Cu

# @jax.jit
# def calculate_phis(A, B, Cx, Cxu, Cu, E):
#     """
#     Same signature/outputs as calculate_phis, but removes the for-loop in (26)
#     using lax.associative_scan to compute prefix products of F_{k,j} = A_k + B_k K_{k,j}.

#     Args:
#       A:   [T,nx,nx]
#       B:   [T,nx,nu]
#       Cx:  [T+1,T,nx,nx]
#       Cxu: [T,T,nx,nu]
#       Cu:  [T,T,nu,nu]
#       E:   [T,nx,nx]

#     Returns:
#       Phi_x: [T+1,T,nx,nx]
#       Phi_u: [T,T,nu,nx]
#     """
#     T, nx, _ = A.shape
#     nu = B.shape[-1]

#     zeros_q = jnp.zeros((T + 1, nx), dtype=A.dtype)
#     zeros_r = jnp.zeros((T, nu), dtype=A.dtype)
#     zeros_c = jnp.zeros((T, nx), dtype=A.dtype)

#     # ---- 1) Riccati (unchanged): solve K_{k,j} for each j in parallel ----
#     def solve_one_j(j):
#         Qj = Cx[:, j, :, :]      # [T+1,nx,nx]
#         Rj = Cu[:, j, :, :]      # [T,nu,nu]
#         Mj = Cxu[:, j, :, :]     # [T,nx,nu]
#         K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
#         return K                 # [T,nu,nx]

#     # K_all: [T(j), T(k), nu, nx] -> swap to K_kj: [T(k), T(j), nu, nx]
#     K_all = jax.vmap(solve_one_j)(jnp.arange(T))
#     K_kj  = jnp.swapaxes(K_all, 0, 1)  # [T, T, nu, nx]

#     # ---- 2) Build F_{k,j} = A_k + B_k K_{k,j}  ----
#     # A[:,None,:,:] broadcasts over j
#     # B[k] @ K[k,j] over j: einsum('x u, j u y -> j x y')
#     BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)          # [T,T,nx,nx]
#     F  = A[:, None, :, :] + BK                          # [T,T,nx,nx]

#     # ---- 3) Turn the lower-triangular recursion into prefix products ----
#     # For each time t (0..T-1) and each j, define element:
#     #   elem[t,j] = I   if t <= j
#     #             = F[t,j] if t > j
#     I = jnp.eye(nx, dtype=A.dtype)
#     t_idx = jnp.arange(T)[:, None]   # [T,1]
#     j_idx = jnp.arange(T)[None, :]   # [1,T]
#     use_F = (t_idx > j_idx)          # [T,T]

#     elems = jnp.where(use_F[:, :, None, None], F, I)  # [T,T,nx,nx]

#     # Prefix product along time:
#     # We want P[t] = elems[t] @ elems[t-1] @ ... @ elems[0] (chronological)
#     # Using associative_scan with compose(l, r) = r @ l achieves that ordering.
#     def compose(l, r):
#         return jnp.einsum("...ab,...bc->...ac", r, l)

#     P = lax.associative_scan(compose, elems, axis=0)  # [T,T,nx,nx]

#     # Now, for k = 1..T:
#     # Phi_x[k,j] = P[k-1,j] @ E[j]
#     # (This gives Phi_x[j+1,j] = I @ E[j] automatically.)
#     Phix_1toT = jnp.einsum("tjab,jbc->tjac", P, E)    # [T,T,nx,nx]
#     Phi_x = jnp.concatenate(
#         [jnp.zeros((1, T, nx, nx), dtype=A.dtype), Phix_1toT],
#         axis=0,
#     )  # [T+1,T,nx,nx]

#     # Mask invalid entries where k <= j  (Phi_x is defined for k >= j+1)
#     k_idx_full = jnp.arange(T + 1)[:, None]              # [T+1,1]
#     valid_x = (k_idx_full >= (j_idx + 1))                # [T+1,T]
#     Phi_x = Phi_x * valid_x[:, :, None, None]

#     # ---- 4) Phi_u[k,j] = K[k,j] @ Phi_x[k,j] for k=0..T-1 ----
#     # Use Phi_x at time k (not k+1), so slice Phi_x[:-1]
#     Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])  # [T,T,nu,nx]

#     k_idx = jnp.arange(T)[:, None]                       # [T,1]
#     valid_u = (k_idx >= (j_idx + 1))                     # [T,T]
#     Phi_u = Phi_u * valid_u[:, :, None, None]

#     return Phi_x, Phi_u

# @jax.jit
# def calculate_phis(A, B, Cx, Cxu, Cu, E):
#     """
#     Corrected to match reference fast_SLS indexing:
#       Phi_x[j,j] = E[j]
#       Phi_u[j,j] = K[j,j] @ Phi_x[j,j]
#       Valid region: k >= j (not k >= j+1)

#     Args:
#       A:   [T,nx,nx]
#       B:   [T,nx,nu]
#       Cx:  [T+1,T,nx,nx]
#       Cxu: [T,T,nx,nu]
#       Cu:  [T,T,nu,nu]
#       E:   [T,nx,nx]  (note: this assumes nw==nx; see note below)

#     Returns:
#       Phi_x: [T+1,T,nx,nx]  where Phi_x[k,j] is defined for k>=j and k<=T
#       Phi_u: [T,T,nu,nx]    where Phi_u[k,j] is defined for k>=j and k<=T-1
#     """
#     T, nx, _ = A.shape
#     nu = B.shape[-1]

#     zeros_q = jnp.zeros((T + 1, nx), dtype=A.dtype)
#     zeros_r = jnp.zeros((T, nu), dtype=A.dtype)
#     zeros_c = jnp.zeros((T, nx), dtype=A.dtype)

#     # ---- 1) Riccati: solve K_{k,j} for each j ----
#     def solve_one_j(j):
#         Qj = Cx[:, j, :, :]      # [T+1,nx,nx]
#         Rj = Cu[:, j, :, :]      # [T,nu,nu]
#         Mj = Cxu[:, j, :, :]     # [T,nx,nu]
#         K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
#         return K                 # [T,nu,nx]

#     K_all = jax.vmap(solve_one_j)(jnp.arange(T))     # [T(j), T(k), nu, nx]
#     K_kj  = jnp.swapaxes(K_all, 0, 1)                # [T(k), T(j), nu, nx]

#     # ---- 2) Closed-loop F[t,j] ----
#     BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)       # [T,T,nx,nx]
#     F  = A[:, None, :, :] + BK                       # [T,T,nx,nx]

#     # ---- 3) Prefix products that start at j (NOT at 0) ----
#     # We want Prod[k,j] = F[k-1,j] @ ... @ F[j,j]  for k>j, and identity for k==j.
#     #
#     # Build elems[t,j] = F[t,j] if t >= j else I
#     I = jnp.eye(nx, dtype=A.dtype)
#     t_idx = jnp.arange(T)[:, None]     # [T,1]
#     j_idx = jnp.arange(T)[None, :]     # [1,T]
#     use_F = (t_idx >= j_idx)           # include t=j
#     elems = jnp.where(use_F[:, :, None, None], F, I)  # [T,T,nx,nx]

#     # associative_scan over t computes:
#     #   P[t] = elems[t] @ elems[t-1] @ ... @ elems[0]
#     # For a given j:
#     #   if t < j: elems = I throughout => P[t,j]=I
#     #   if t >= j: P[t,j] = F[t,j] @ ... @ F[j,j]
#     def compose(l, r):
#         return jnp.einsum("...ab,...bc->...ac", r, l)

#     P = lax.associative_scan(compose, elems, axis=0)  # [T,T,nx,nx]

#     # Now map to Phi_x:
#     #   Phi_x[j,j]   = E[j]
#     #   Phi_x[k,j]   = P[k-1,j] @ E[j]   for k = 1..T
#     # BUT: this should be valid for k>=j.
#     Phix_1toT = jnp.einsum("tjab,jbc->tjac", P, E)    # [T,T,nx,nx] corresponds to k=t+1
#     Phi_x = jnp.concatenate(
#         [jnp.zeros((1, T, nx, nx), dtype=A.dtype), Phix_1toT],
#         axis=0,
#     )  # [T+1,T,nx,nx]

#     # Overwrite diagonal Phi_x[j,j] with E[j] explicitly.
#     # (This also fixes k=0,j=0 case properly.)
#     Phi_x = Phi_x.at[jnp.arange(T), jnp.arange(T)].set(E)

#     # Mask: Phi_x[k,j] valid for k >= j (reference includes k=j)
#     k_idx_full = jnp.arange(T + 1)[:, None]   # [T+1,1]
#     valid_x = (k_idx_full >= j_idx)           # [T+1,T]
#     Phi_x = Phi_x * valid_x[:, :, None, None]

#     # ---- 4) Phi_u[k,j] = K[k,j] @ Phi_x[k,j] for k=0..T-1, valid for k>=j ----
#     Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])  # [T,T,nu,nx]

#     k_idx = jnp.arange(T)[:, None]     # [T,1]
#     valid_u = (k_idx >= j_idx)         # include diagonal k=j
#     Phi_u = Phi_u * valid_u[:, :, None, None]

#     return Phi_x, Phi_u

import jax
import jax.numpy as jnp
from jax import lax

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
def get_controller(Q, R, A, B, C, D, E, eta):
    """
    Controller optimization step consistent with Fast-SLS Theorem 3, assuming all inputs are LTV.

    Args:
      Q:   [T+1,nx,nx]
      R:   [T,nu,nu]
      A:   [T,nx,nx]
      B:   [T,nx,nu]
      C:   [T,nc,nx]
      D:   [T,nc,nu]
      E:   [T,nx,nx]
      eta: [T,T,nc]  (eta[k,j,i] >= 0)

    Returns:
      Phi_x: [T+1,T,nx,nx]
      Phi_u: [T,T,nu,nx]
      (Cx, Cxu, Cu): cost blocks used in Riccati
    """
    T, nx, _ = A.shape

    # Build per-(k,j) cost blocks for k=0..T-1
    js = jnp.arange(T)
    ks = jnp.arange(T)

    def blocks_for_k(k):
        def blocks_for_j(j):
            return calculate_cost(Q[k], R[k], C[k], D[k], eta[k, j])
        return vmap(blocks_for_j)(js)

    Cx_kj, Cxu_kj, Cu_kj = vmap(blocks_for_k)(ks)
    # Shapes:
    #   Cx_kj:  [T,T,nx,nx]
    #   Cxu_kj: [T,T,nx,nu]
    #   Cu_kj:  [T,T,nu,nu]

    # Terminal Cx_N,j: if you have terminal constraint rows, incorporate them similarly.
    # Here: terminal cost uses Q[T] (interpretable as P_bar / terminal state penalty).
    Cx_Nj = jnp.broadcast_to(Q[T], (T, nx, nx))              # [T,nx,nx]
    Cx = jnp.concatenate([Cx_kj, Cx_Nj[None, ...]], axis=0)  # [T+1,T,nx,nx]
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
    nc = C.shape[1]

    # Stage: k=0..T-1
    # term_x[k,j,i,:] = C[k,i] @ Phi_x[k,j]
    term_x = jnp.einsum("kix,kjxn->kjin", C[:-1], Phi_x[:-1])  # [T, T+1, nc, nw]
    # term_u[k,j,i,:] = D[k,i] @ Phi_u[k,j]
    term_u = jnp.einsum("kiu,kjun->kjin", D[:-1], Phi_u)            # [T, T+1, nc, nw]
    gPhi = term_x + term_u
    beta_stage = jnp.sum(gPhi * gPhi, axis=-1)                 # [T, T+1, nc]

    # Mask invalid entries (j > k) to 0 to keep your triangular structure
    k_idx = jnp.arange(T)[:, None]          # [T,1]
    j_idx = jnp.arange(Tp1)[None, :]        # [1,T+1]
    mask = (j_idx <= k_idx)                 # [T, T+1]
    beta_stage = beta_stage * mask[:, :, None]

    # Terminal row (k=T): only state constraints contribute (no D at terminal)
    # beta_term[j,i] = || C[T,i] @ Phi_x[T,j] ||^2
    gPhi_term = jnp.einsum("ix,jxn->jin", C[-1], Phi_x[-1])    # [T+1, nc, nw]
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
    h_ct = h_ct + eps_beta
    h_ct = h_ct.at[0].set(jnp.zeros((nc,), dtype=h_ct.dtype))
    return h_ct


@jax.jit
def get_etas(mus, betas, eps=1e-12):
    """
    mus:   [T+1, nc]
    betas: [T+1, T+1, nc]
    returns eta: [T, T, nc] (uses only j=0..T-1)
    """
    mu_k   = mus[:-1]                 # [T, nc]
    beta_k = betas[:-1, :-1]          # [T, T, nc]  <-- FIX

    mu_k = mu_k[:, None, :]           # [T,1,nc]
    eta = mu_k / (2.0 * jnp.sqrt(jnp.maximum(beta_k, eps)))

    T = eta.shape[0]
    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(T)[None, :]
    mask = (k_idx >= j_idx)
    eta = eta * mask[:, :, None]
    eta = jnp.maximum(eta, 0.0)
    return eta

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
                       sls_config: SLSConfig, E: jnp.ndarray):
    # Solve Nominal Trajectory
    
    Tp1 = Q.shape[0]
    nx  = Q.shape[1]
    nu  = R.shape[1]
    nc  = w.shape[1]
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
    carry0 = (i0, beta0, x0, u0, v0, w, y, rho, converged0, converged0, h_ct0)

    def cond_fn(carry):
        i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _, _ = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_fn(carry):
        i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _, _ = carry

        prev_rho = rho
        x_prev = x_curr
        u_prev = u_curr

        h_ct = get_constraint_tightenings(beta)
        tightened_constraints = f - h_ct
        # w = jnp.zeros_like(w)
        # y = jnp.zeros_like(y)
        # rho = jnp.minimum(0.1, rho)
        x_curr, u_curr, v_curr, w, y, rho, mu, converged_admm = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C, D, tightened_constraints, w, y, rho
        )

        metric = primal_convergence_metric(x_curr, u_curr, x_prev, u_prev)
        eta = get_etas(mu, beta)
        Phi_x, Phi_u = get_controller(Q, R, A, B, C, D, E, eta)
        beta = get_betas(C, D, Phi_x, Phi_u)

        rho = jnp.maximum(jnp.minimum(rho, 1e3) * 0.5, 0.1)
        y = prev_rho / rho * y

        converged_now = metric <= tol
        converged = jnp.logical_or(converged, converged_now)

        return (i + jnp.array(1, dtype=jnp.int32),
                beta, x_curr, u_curr, v_curr, w, y, rho, converged, converged_admm, h_ct)

    carryN = jax.lax.while_loop(cond_fn, body_fn, carry0)

    _, betaN, xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct = carryN
    return xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct