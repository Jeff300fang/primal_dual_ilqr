from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax, vmap
from mpx.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import tvlqr_gpu
import time

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

@jax.jit
def calculate_phis(A, B, Cx, Cxu, Cu, E):
    """
    Same signature/outputs as calculate_phis, but removes the for-loop in (26)
    using lax.associative_scan to compute prefix products of F_{k,j} = A_k + B_k K_{k,j}.

    Args:
      A:   [T,nx,nx]
      B:   [T,nx,nu]
      Cx:  [T+1,T,nx,nx]
      Cxu: [T,T,nx,nu]
      Cu:  [T,T,nu,nu]
      E:   [T,nx,nx]

    Returns:
      Phi_x: [T+1,T,nx,nx]
      Phi_u: [T,T,nu,nx]
    """
    T, nx, _ = A.shape
    nu = B.shape[-1]

    zeros_q = jnp.zeros((T + 1, nx), dtype=A.dtype)
    zeros_r = jnp.zeros((T, nu), dtype=A.dtype)
    zeros_c = jnp.zeros((T, nx), dtype=A.dtype)

    # ---- 1) Riccati (unchanged): solve K_{k,j} for each j in parallel ----
    def solve_one_j(j):
        Qj = Cx[:, j, :, :]      # [T+1,nx,nx]
        Rj = Cu[:, j, :, :]      # [T,nu,nu]
        Mj = Cxu[:, j, :, :]     # [T,nx,nu]
        K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
        return K                 # [T,nu,nx]

    # K_all: [T(j), T(k), nu, nx] -> swap to K_kj: [T(k), T(j), nu, nx]
    K_all = jax.vmap(solve_one_j)(jnp.arange(T))
    K_kj  = jnp.swapaxes(K_all, 0, 1)  # [T, T, nu, nx]

    # ---- 2) Build F_{k,j} = A_k + B_k K_{k,j}  ----
    # A[:,None,:,:] broadcasts over j
    # B[k] @ K[k,j] over j: einsum('x u, j u y -> j x y')
    BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)          # [T,T,nx,nx]
    F  = A[:, None, :, :] + BK                          # [T,T,nx,nx]

    # ---- 3) Turn the lower-triangular recursion into prefix products ----
    # For each time t (0..T-1) and each j, define element:
    #   elem[t,j] = I   if t <= j
    #             = F[t,j] if t > j
    I = jnp.eye(nx, dtype=A.dtype)
    t_idx = jnp.arange(T)[:, None]   # [T,1]
    j_idx = jnp.arange(T)[None, :]   # [1,T]
    use_F = (t_idx > j_idx)          # [T,T]

    elems = jnp.where(use_F[:, :, None, None], F, I)  # [T,T,nx,nx]

    # Prefix product along time:
    # We want P[t] = elems[t] @ elems[t-1] @ ... @ elems[0] (chronological)
    # Using associative_scan with compose(l, r) = r @ l achieves that ordering.
    def compose(l, r):
        return jnp.einsum("...ab,...bc->...ac", r, l)

    P = lax.associative_scan(compose, elems, axis=0)  # [T,T,nx,nx]

    # Now, for k = 1..T:
    # Phi_x[k,j] = P[k-1,j] @ E[j]
    # (This gives Phi_x[j+1,j] = I @ E[j] automatically.)
    Phix_1toT = jnp.einsum("tjab,jbc->tjac", P, E)    # [T,T,nx,nx]
    Phi_x = jnp.concatenate(
        [jnp.zeros((1, T, nx, nx), dtype=A.dtype), Phix_1toT],
        axis=0,
    )  # [T+1,T,nx,nx]

    # Mask invalid entries where k <= j  (Phi_x is defined for k >= j+1)
    k_idx_full = jnp.arange(T + 1)[:, None]              # [T+1,1]
    valid_x = (k_idx_full >= (j_idx + 1))                # [T+1,T]
    Phi_x = Phi_x * valid_x[:, :, None, None]

    # ---- 4) Phi_u[k,j] = K[k,j] @ Phi_x[k,j] for k=0..T-1 ----
    # Use Phi_x at time k (not k+1), so slice Phi_x[:-1]
    Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])  # [T,T,nu,nx]

    k_idx = jnp.arange(T)[:, None]                       # [T,1]
    valid_u = (k_idx >= (j_idx + 1))                     # [T,T]
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
    C: [T+1,nc,nx]  (includes terminal x_T constraints)
    D: [T,  nc,nu]  (stage controls only)
    """
    T = Phi_u.shape[0]
    nu = Phi_u.shape[2]
    nx = Phi_u.shape[3]

    Phi_u_pad = jnp.concatenate(
        [Phi_u, jnp.zeros((1, T, nu, nx), dtype=Phi_u.dtype)],
        axis=0,
    )
    # stage gPhi[k,j,i,:] = C[k,i] Phi_x[k,j] + D[k,i] Phi_u[k,j]
    term_x = jnp.einsum("kix,kjxy->kjiy", C, Phi_x)  # [T,T,nc,nx]
    term_u = jnp.einsum("kiu,kjux->kjix", D, Phi_u_pad)       # [T,T,nc,nx]
    gPhi = term_x + term_u
    beta = jnp.sum(gPhi * gPhi, axis=-1)
    return beta

@jax.jit
def get_constraint_tightenings(betas, eps_beta=1e-6):
    """
    betas: [T+1, T, nc] with betas[k,j,i] = ||g_{k,i}^T Phi_{k,j}||^2
    p:     scalar (or [T+1,1] or [T+1,nc] if you want it time/constraint-varying)
    eps_beta: scalar >= 0 (floor term)

    Returns:
      h_ct: [T+1, nc]
        h_ct[0] = 0
        h_ct[k] = sum_{j=0}^{k-1} p * sqrt(betas[k,j,:]) + eps_beta
    """
    T1, T, nc = betas.shape  # T1 = T+1

    # sqrt(beta) is what shows up in your definition
    s = jnp.sqrt(jnp.maximum(betas, 0.0))  # [T+1, T, nc]

    # Mask to enforce j < k
    k_idx = jnp.arange(T1)[:, None]   # [T+1,1]
    j_idx = jnp.arange(T)[None, :]    # [1,T]
    valid = (j_idx < k_idx)           # [T+1,T]
    s = s * valid[:, :, None]

    # Sum over j
    h_ct = jnp.sum(s, axis=1)  # [T+1, nc]

    h_ct = h_ct + eps_beta
    h_ct = h_ct.at[0].set(jnp.zeros((nc,), dtype=h_ct.dtype))
    return h_ct

@jax.jit
def get_etas(mus, betas, eps=1e-12):
    """
    Args:
      mus:   [T+1, nc]
      betas: [T+1, T, nc]   (beta_bar)

    Returns:
      eta:   [T, T, nc]
    """
    # Stage-only k = 0..T-1
    mu_k   = mus[:-1]            # [T, nc]
    beta_k = betas[:-1]          # [T, T, nc]

    # Broadcast mu_k over j
    mu_k = mu_k[:, None, :]      # [T, 1, nc]

    # Core formula
    eta = mu_k / (2.0 * jnp.sqrt(jnp.maximum(beta_k, eps)))

    # Enforce j <= k (lower triangular, including diagonal)
    T = eta.shape[0]
    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(T)[None, :]
    mask = (k_idx >= j_idx)

    eta = eta * mask[:, :, None]

    # Numerical safety
    eta = jnp.maximum(eta, 0.0)
    return eta

def make_dummy_problem(
    T=5,
    nx=4,
    nu=2,
    nc=3,
    seed=0,
    dtype=jnp.float32,
):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)

    # -------------------------
    # Dynamics: x_{k+1} = A_k x_k + B_k u_k
    # -------------------------
    A = []
    B = []
    for k in range(T):
        Ak = (
            0.95 * jnp.eye(nx, dtype=dtype)
            + 0.05 * jax.random.normal(keys[0], (nx, nx), dtype=dtype)
        )
        Bk = 0.1 * jax.random.normal(keys[1], (nx, nu), dtype=dtype)
        A.append(Ak)
        B.append(Bk)

    A = jnp.stack(A)  # [T,nx,nx]
    B = jnp.stack(B)  # [T,nx,nu]

    # -------------------------
    # Cost: quadratic LQR base
    # -------------------------
    Q = []
    for k in range(T + 1):
        Qk = jnp.eye(nx, dtype=dtype)
        Q.append(Qk)
    Q = jnp.stack(Q)  # [T+1,nx,nx]

    R = []
    for k in range(T):
        Rk = 0.1 * jnp.eye(nu, dtype=dtype)
        R.append(Rk)
    R = jnp.stack(R)  # [T,nu,nu]

    # -------------------------
    # Constraint Jacobians
    #   g_k(x,u) = C_k x + D_k u
    # -------------------------
    C = []
    D = []
    for k in range(T + 1):
        Ck = jax.random.normal(keys[2], (nc, nx), dtype=dtype)
        Dk = jax.random.normal(keys[3], (nc, nu), dtype=dtype)
        C.append(Ck)
        D.append(Dk)

    C = jnp.stack(C)  # [T,nc,nx]
    D = jnp.stack(D)  # [T,nc,nu]

    # -------------------------
    # Disturbance injection matrix E
    # Phi_x[j+1,j] = E[j]
    # -------------------------
    E = []
    for k in range(T):
        Ek = 0.1 * jnp.eye(nx, dtype=dtype)
        E.append(Ek)
    E = jnp.stack(E)  # [T,nx,nx]

    # -------------------------
    # Dual weights eta[k,j,i] >= 0
    # -------------------------
    eta = jax.random.uniform(
        keys[4],
        shape=(T, T, nc),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    mu_time = jnp.linspace(1.0, 2.0, T + 1, dtype=dtype)
    key = jax.random.PRNGKey(0)
    mus = 0.5 + jax.random.uniform(key, (T+1, nc), dtype=dtype)
    return Q, R, A, B, C, D, E, eta, mus

def main():
    Q, R, A, B, C, D, E, eta, mus = make_dummy_problem(T=50, nx=61, nu=13, nc=10)
    Phi_x, Phi_u = get_controller(
        Q=Q,
        R=R,
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        eta=eta,
    )
    betas = get_betas(C, D, Phi_x, Phi_u)
    h_ct  = get_constraint_tightenings(betas, eps_beta=1e-6)
    eta = get_etas(mus, betas)
    start = time.perf_counter()
    Phi_x, Phi_u = get_controller(
        Q=Q,
        R=R,
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        eta=eta,
    )
    betas = get_betas(C, D, Phi_x, Phi_u)
    print(betas.shape)
    h_ct  = get_constraint_tightenings(betas, eps_beta=1e-6)
    eta = get_etas(mus, betas)
    end = time.perf_counter()
    print(end - start)
    print(h_ct.shape)


if __name__ == '__main__':
    main()