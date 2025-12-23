from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax, vmap
from mpx.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import tvlqr_gpu
import time

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

def calculate_phis(A, B, Cx, Cxu, Cu, E):
    """
    Theorem 3:
      - N parallel Riccati recursions (via tvlqr_gpu) to get K_{k,j}
      - N parallel forward propagations to get Phi_x, Phi_u

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

    # Solve one Riccati recursion per disturbance index j
    # Q_j[t]=Cx[t,j], R_j[t]=Cu[t,j], M_j[t]=Cxu[t,j]
    def solve_one_j(j):
        Qj = Cx[:, j, :, :]     # [T+1,nx,nx]
        Rj = Cu[:, j, :, :]     # [T,nu,nu]
        Mj = Cxu[:, j, :, :]    # [T,nx,nu]
        K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
        return K                # [T,nu,nx]

    # K_all has axes (j,k,nu,nx)
    K_all = vmap(solve_one_j)(jnp.arange(T))  # [T, T, nu, nx]
    # Convert to K[k,j]
    K_kj = jnp.swapaxes(K_all, 0, 1)          # [T(k), T(j), nu, nx]

    # Forward propagation (26)
    Phi_x = jnp.zeros((T + 1, T, nx, nx), dtype=A.dtype)
    Phi_u = jnp.zeros((T, T, nu, nx), dtype=A.dtype)

    # Phi_x[j+1,j] = E[j]
    Phi_x = Phi_x.at[jnp.arange(T) + 1, jnp.arange(T), :, :].set(E)

    def body(k, carry):
        Phi_x, Phi_u = carry

        # Current Phi_x[k,j] over all j
        Phix_kj = Phi_x[k, :, :, :]          # [T, nx, nx]
        K_kj_k  = K_kj[k, :, :, :]           # [T, nu, nx]

        # Phi_u[k,j] = K[k,j] @ Phi_x[k,j]
        Phiu_kj = jnp.einsum("jmn,jnp->jmp", K_kj_k, Phix_kj)  # [T, nu, nx]

        # Phi_x[k+1,j] = (A[k] + B[k]K[k,j]) Phi_x[k,j]
        AK = A[k] + jnp.einsum("nm,jmp->jnp", B[k], K_kj_k)    # [T, nx, nx]
        Phix_next = jnp.einsum("jnn,jnp->jnp", AK, Phix_kj)    # [T, nx, nx]

        # Valid only for j <= k-1 (since Phi_x[k,j] is defined for k>=j+1)
        valid = (k > jnp.arange(T))[:, None, None]  # [T,1,1]

        Phi_u = Phi_u.at[k, :, :, :].set(Phiu_kj * valid)
        Phi_x = Phi_x.at[k + 1, :, :, :].set(Phi_x[k + 1] + Phix_next * valid)

        return (Phi_x, Phi_u)

    Phi_x, Phi_u = lax.fori_loop(1, T, body, (Phi_x, Phi_u))
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
    Phi_x_p, Phi_u_p = calculate_phis_parallel(A, B, Cx, Cxu_kj, Cu_kj, E)
    return Phi_x, Phi_u, (Cx, Cxu_kj, Cu_kj)

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
    for k in range(T):
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

    return Q, R, A, B, C, D, E, eta

def main():
    Q, R, A, B, C, D, E, eta = make_dummy_problem(T=200, nx=61, nu=13, nc=10)
    get_controller(
        Q=Q,
        R=R,
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        eta=eta,
    )
    start = time.perf_counter()
    get_controller(
        Q=Q,
        R=R,
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        eta=eta,
    )
    end = time.perf_counter()
    print(end - start)


if __name__ == '__main__':
    main()