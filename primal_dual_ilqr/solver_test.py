import time
import jax.numpy as jnp
import jax
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_pcr import ADMMProblem, ADMMConfig, ADMMPCR

def main():
    nx_single = 2
    nu_single = 1
    nc_single = 6
    num_blocks = 40
    N = 24

    nx = nx_single * num_blocks
    nu = nu_single * num_blocks
    nc = nc_single * num_blocks
    nc = 0

    Q_block = jnp.diag(jnp.array([100.0, 20.0]))
    R_block = jnp.diag(jnp.array([10.0]))
    A_block = jnp.array([[1.0, 0.5],
                         [0.0, 1.0]])
    B_block = jnp.array([[0.125],
                         [0.5]])
    
    c_block = jnp.array([0.0,
                        0.0])

    C_block = jnp.array([
        [-1.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0, -1.0],
        [ 0.0,  1.0],
        [ 0.0,  0.0],
        [ 0.0,  0.0],
    ])
    D_block = jnp.array([
        [ 0.0],
        [ 0.0],
        [ 0.0],
        [ 0.0],
        [ 1.0],
        [-1.0],
    ])

    f_block = jnp.array([[5.0], [5.0], [5.0], [5.0], [1.0], [1.0]])

    # Initial and terminal states
    x0 = jnp.ones(nx) * -5.0
    # set all odd indices (velocities) to 0
    x0 = x0.at[1::2].set(0.0)

    xN = jnp.ones(nx) * 5.0
    xN = xN.at[1::2].set(0.0)

    # Preallocate
    Q = jnp.zeros((N + 1, nx, nx))
    q = jnp.zeros((N + 1, nx))
    R = jnp.zeros((N, nu, nu))
    r = jnp.zeros((N, nu))
    A = jnp.zeros((N, nx, nx))
    B = jnp.zeros((N, nx, nu))
    C = jnp.zeros((N + 1, nc, nx))
    c = jnp.zeros((N, nx))
    D = jnp.zeros((N, nc, nu))
    f = jnp.zeros((N + 1, nc, 1))

    # Build per-stage matrices
    for k in range(N):
        # Dynamics & cost
        A_k = jnp.kron(jnp.eye(num_blocks), A_block)
        B_k = jnp.kron(jnp.eye(num_blocks), B_block)
        c_k = jnp.kron(jnp.ones((num_blocks,), dtype=jnp.float32), c_block)
        C_k = jnp.kron(jnp.eye(num_blocks), C_block)
        D_k = jnp.kron(jnp.eye(num_blocks), D_block)
        Q_k = jnp.kron(jnp.eye(num_blocks), Q_block)
        R_k = jnp.kron(jnp.eye(num_blocks), R_block)
        f_k = jnp.kron(jnp.ones((num_blocks, 1)), f_block)

        A = A.at[k].set(A_k)
        B = B.at[k].set(B_k)
        c = c.at[k].set(c_k)
        C = C.at[k].set(C_k)
        D = D.at[k].set(D_k)
        Q = Q.at[k].set(Q_k)
        R = R.at[k].set(R_k)
        f = f.at[k].set(f_k)

        q_k = -Q_k @ xN
        q = q.at[k].set(q_k)

    # terminal cost / constraints
    Q_N = jnp.kron(jnp.eye(num_blocks), Q_block)
    C_N = jnp.kron(jnp.eye(num_blocks), C_block)
    f_N = jnp.kron(jnp.ones((num_blocks, 1)), f_block)

    Q = Q.at[N].set(Q_N)
    q = q.at[N].set(-Q_N @ xN)
    C = C.at[N].set(C_N)
    f = f.at[N].set(f_N)
    C = jnp.zeros((N + 1, nc, nx), dtype=Q.dtype)          # (T+1,0,nx)
    D = jnp.zeros((N + 1, nc, nu), dtype=Q.dtype)          # (T+1,0,nu)
    f = jnp.zeros((N + 1, nc, 1), dtype=Q.dtype)           # (T+1,0,1)
    nc_single = 0
    problem = ADMMProblem(Q, q, R, r, A, B, c, C, D, f, N, nx_single * num_blocks, nu_single * num_blocks, nc_single * num_blocks)

    config = ADMMConfig(
        max_iterations=400,
        rho_update_frequency=25,
        condense_block_size=6,
        eps_abs=1e-6,
        eps_rel=1e-4,
    )
    
    # Change condense_block_size to e.g. 2, 3, 5 to enable partial condensing
    solver = ADMMPCR(
        config
    )

    # Run one just compile
    x_traj, u_traj = solver.solve(problem, x0, warm_start=False)  # expect JAX arrays

    logdir = "/tmp/jax_prof"

    # profiler.start_trace(logdir)
    
    print("start")
    total = 0
    sim = 1
    for i in range(sim):
        start = time.perf_counter()
        x_traj, u_traj = solver.solve(problem, x0, warm_start=False)  # expect JAX arrays
        end = time.perf_counter()
        total += (end - start)
    # print(total / sim * 1000, "ms")

    # profiler.stop_trace()

    # Move to host for printing (if needed)
    x_traj_host = jax.device_get(x_traj)
    u_traj_host = jax.device_get(u_traj)

    for elem, vel, u in zip(x_traj_host[:, 0], x_traj_host[:, 1], u_traj_host[:, 0]):
        print("State:", float(elem), "Velocity:", float(vel), "Control:", float(u))

    print("State:", float(x_traj_host[N, 0]), "Velocity:", float(x_traj_host[N, 1]))

    # ------- Tests Dynamics Satisfaction -------
    for k in range(N):
        xk = x_traj_host[k, :2]
        uk = u_traj_host[k, 0]

        x_next_pred = A_block @ xk + B_block.flatten() * uk + c_block
        diff = jnp.linalg.norm(x_traj_host[k + 1, :2] - x_next_pred)
        diff_val = float(diff)
        print(k, diff_val)
        if diff_val > 1e-12:
            print("Dynamics not satisfied!")
            break
    # -------------------------------------------

    print(f"Solve time: {(end - start) * 1000} ms",)
    print(total / sim * 1000, "ms")


if __name__ == '__main__':
    main()
