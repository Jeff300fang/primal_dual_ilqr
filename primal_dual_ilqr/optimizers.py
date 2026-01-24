from jax import debug, grad, jit, lax, scipy, vmap
import jax
import jax.numpy as jnp

from functools import partial

from trajax.optimizers import linearize, quadratize,vectorize
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import fast_sls_solve_gpu
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import constrained_solve
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass

@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True

    def tree_flatten(self):
        children = (self.max_sqp_iterations, self.feas_tol, self.step_tol, self.warm_start)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

def linearize_scan(fun, argnums=3):
    """Gradient or Jacobian operator using scan.

    Args:
        fun: numpy scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to process.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory.

        Example:
            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically jnp.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def scan_fun(carry, inputs):
        args = (*carry, *inputs)
        A = jacobian_x(*args)
        B = jacobian_u(*args)
        return carry, (A, B)

    def linearizer(x, u, t, *args):
        inputs = (x, u, t)
        _, (A, B) = lax.scan(scan_fun, args, inputs)
        return A, B

    return linearizer
def linearize_obj_scan(fun, argnums=5):
    """Gradient or Jacobian operator using scan.

    Args:
        fun: numpy scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to process.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory.

        Example:
            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically jnp.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def scan_fun(carry, inputs):
        args = (*carry, *inputs)
        A = jacobian_x(*args)
        B = jacobian_u(*args)
        return carry, (A, B)

    def linearizer(x, u,  v, v1, t, *args):
        inputs = (x, u, v, v1,t)
        _, (A, B) = lax.scan(scan_fun, args, inputs)
        return A, B

    return linearizer
def lagrangian(cost, dynamics, x0):
    """Returns a function to evaluate the associated Lagrangian."""

    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun

@jax.jit
def add_obstacle_constraints(C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                             obstacles: jnp.ndarray, x_curr: jnp.ndarray, eps=1e-5):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _,  _, nu = D.shape

    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]

    pos = x_curr[:, :2]

    diff = pos[:, None, :] - centers[None, :, :]

    dist = jnp.linalg.norm(diff, axis=-1) + eps

    n = diff / dist[..., None]

    coeffs = -n

    C_obstacle = jnp.zeros((Tp1, centers.shape[0], nx), dtype=C.dtype)
    D_obstacle = jnp.zeros((Tp1, centers.shape[0], nu), dtype=D.dtype)

    C_obstacle = C_obstacle.at[..., 0:2].set(coeffs)

    f_obstacle = (dist - radii[None, :]).astype(f.dtype)

    C_all = jnp.concatenate([C, C_obstacle], axis=1)
    D_all = jnp.concatenate([D, D_obstacle], axis=1)
    f_all = jnp.concatenate([f, f_obstacle], axis=1)
    return C_all, D_all, f_all


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7))
def compute_search_direction(
    sls_config: SLSConfig,
    admm_config,
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    constraints, disturbance, obstacles,
    x0,
    X,
    U,
    V,
    c,
    w, y, rho,
    h_ct_ws,
):
    """Computes the SQP search direction.

    Args:
      cost:          cost function with signature cost(x, u, t).
      dynamics:      dynamics function with signature dynamics(x, u, t).
      x0:            [n]           numpy array.
      X:             [T+1, n]      numpy array.
      U:             [T, m]        numpy array.
      V:             [T+1, n]      numpy array.
      c:             [T+1, n]      numpy array.
      make_psd:      whether to zero negative eigenvalues after quadratization.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      q: [T+1, n]  numpy array.
      r: [T, m]    numpy array.
    """
    T = U.shape[0]
    nc = w.shape[1]
    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), jnp.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), jnp.arange(T + 1))

    R = R_pad[:-1]
    M = M_pad[:-1]

    linearizer = linearize(lagrangian(cost, dynamics, x0),argnums = 5)
    dynamics_linearizer = linearize(dynamics)

    q, r_pad = linearizer(X, pad(U), jnp.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]

    A_pad, B_pad = dynamics_linearizer(X, pad(U), jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]
    nx = A.shape[1]
    nu = B.shape[2]

    pad = lambda A: jnp.pad(A, ((0, 1), (0, 0)))  # (T,m) -> (T+1,m)
    U_pad = pad(U)
    t = jnp.arange(X.shape[0])  # (T+1,)

    g = vectorize(constraints)(X, U_pad, t)
    f = -g

    C, D = linearize(constraints)(X, U_pad, t)

    C_all, D_all, f_all = add_obstacle_constraints(C, D, f, obstacles, X)

    E = disturbance(X)
    cfg = admm_config

    # Solve constrained QP for the SQP step (dX, dU)
    # TODO: Correctly set Q_bar and R_bar?
    Q_bar = jnp.broadcast_to(jnp.eye(Q.shape[1]), Q.shape)
    R_bar = jnp.broadcast_to(jnp.eye(R.shape[1]), R.shape)
    # Q_bar = Q
    # R_bar = R
    if sls_config.enable_fastsls:
        dX, dU, dV, w, y, rho, converged, converged_admm, backoffs, Phi_x, Phi_u = fast_sls_solve_gpu(
            cfg, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho, sls_config, E, Q_bar, R_bar, obstacles, X, h_ct_ws,
        )
    else:
        dX, dU, dV, w, y, rho, _, converged_admm = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho
        )
        converged = True
        backoffs = jnp.zeros((T + 1, nc - obstacles.shape[0]))
        Phi_x = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u = jnp.zeros((T, T + 1, nu, nx))

    # def converged_branch(state):
    #     # state = (w, y, rho)
    #     return dX, dU, dV, state[0], state[1], state[2], converged, converged_admm, backoffs, Phi_x, Phi_u

    # def not_converged_branch(state):
    #     return dX, dU, dV, state[0], state[1], state[2], converged, converged_admm, backoffs, Phi_x, Phi_u
    #     w0, y0, rho0 = state
    #     w_init = jnp.zeros_like(w0)
    #     y_init = jnp.zeros_like(y0)
    #     rho_init = jnp.asarray(0.1, dtype=rho0.dtype)
    #     if sls_config.enable_fastsls:
    #         dX2, dU2, dV2, w2, y2, rho2, conv2, converged_admm2, backoffs, Phi_x, Phi_u = fast_sls_solve_gpu(
    #             cfg, Q, q, R, r, M, A, B, c, C, D, f, w_init, y_init, rho_init,
    #             sls_config, E, Q, R
    #         )
    #     else:
    #         dX2, dU2, dV2, w2, y2, rho2, mu, converged_admm2 = constrained_solve(
    #             cfg, Q, q, R, r, M, A, B, c, C, D, f, w_init, y_init, rho_init
    #         )
    #         conv2 = True
    #         backoffs = jnp.zeros((T + 1, nc))
    #         Phi_x = jnp.zeros((T + 1, T + 1, nx, nx))
    #         Phi_u = jnp.zeros((T, T + 1, nu, nx))
    #     return dX2, dU2, dV2, w2, y2, rho2, conv2, converged_admm2, backoffs, Phi_x, Phi_u

    # dX, dU, dV, w, y, rho, converged, converged_admm, backoffs, Phi_x, Phi_u = lax.cond(
    #     converged_admm,
    #     converged_branch,
    #     not_converged_branch,
    #     operand=(w, y, rho),
    # )

    return dX, dU, dV, q, r, w, y, rho, backoffs, Phi_x, Phi_u

@jit
def merit_rho(c, dV):
    """Determines the merit function penalty parameter to be used.

    Args:
      c:             [T+1, n]  numpy array.
      dV:            [T+1, n]  numpy array.

    Returns:
        rho: the penalty parameter.
    """
    c2 = jnp.sum(c * c)
    dV2 = jnp.sum(dV * dV)
    return lax.select(c2 > 1e-12, 2.0 * jnp.sqrt(dV2 / c2), 1e-2)


@jit
def slope(dX, dU, dV, c, q, r, rho):
    """Determines the directional derivative of the merit function.

    Args:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      c:  [T+1, n] numpy array.
      q:  [T+1, n] numpy array.
      r:  [T, m] numpy array.
      rho: the penalty parameter of the merit function.

    Returns:
        dir_derivative: the directional derivative.
    """
    return jnp.sum(q * dX) + jnp.sum(r * dU) + 2*jnp.sum(dV * c) - rho * jnp.sum(c * c)

@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
    alpha_0,
    alpha_mult,
    alpha_min,
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        # debug.print(f"{new_merit=}, {current_merit=}, {alpha=}, {merit_slope=}")\
        return jnp.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha *= alpha_mult
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = jnp.where(jnp.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, new_merit, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, jnp.inf, alpha_0 / alpha_mult),
    )
    no_errors = alpha > alpha_min


    return X, U, V, new_g, new_c, no_errors

def merit_function_factory(rho_merit):
    def merit_fn(V, g, c):
        return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c)
    return merit_fn

@partial(jit, static_argnums=(0, 1))
def parallel_line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
):
    # Candidate step sizes: 1, 1/2, 1/4, ..., 1/1024
    alpha_values = jnp.exp2(-jnp.arange(11))

    def trial(alpha):
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = jnp.where(jnp.isnan(new_merit), jnp.inf, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit

    Xc, Uc, Vc, gc, cc, mc = vmap(trial)(alpha_values)

    # Armijo sufficient decrease condition:
    # accept if new_merit <= current_merit + alpha * c1 * merit_slope
    rhs = current_merit + alpha_values * armijo_factor * merit_slope
    accepted = mc <= rhs

    # Pick the *largest* alpha that satisfies Armijo. Since alpha_values is descending,
    # the first True is best.
    any_acc = jnp.any(accepted)
    best_index = jnp.where(any_acc, jnp.argmax(accepted), alpha_values.shape[0] - 1)

    return Xc[best_index], Uc[best_index], Vc[best_index], gc[best_index], cc[best_index]

@partial(jit, static_argnums=(0))
def filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_cost,
    current_c,
    q,
    r,
    # Hyperparameters
    alpha_min=1e-4,
    theta_max=1e-2,
    theta_min=1e-6,
    eta=1e-4,
    gamma_phi=1e-6,
    gamma_theta=1e-6,
    gamma_alpha=0.5,
):
    """Performs a backtracking line search.

    Args:
      X_in: [T+1, n] numpy array of current states.
      U_in: [T, m] numpy array of current controls.
      V_in: [T+1, n] numpy array of current multipliers.
      dX: [T+1, n] numpy array of state search direction.
      dU: [T, m] numpy array of control search direction.
      dV: [T+1, n] numpy array of multiplier search direction.
      current_cost: Current cost value (phi_k).
      current_c: Current constraint violation (theta_k).
      alpha_min: Minimum step size.
      theta_max: Maximum acceptable constraint violation.
      theta_min: Minimum constraint violation worth considering.
      eta: Armijo parameter for sufficient decrease.
      gamma_phi: Cost reduction parameter.
      gamma_theta: Constraint violation reduction parameter.
      gamma_alpha: Step size reduction factor.

    Returns:
      X: [T+1, n] numpy array of updated states.
      U: [T, m] numpy array of updated controls.
      V: [T+1, n] numpy array of updated multipliers.
      new_cost: Updated cost value.
      new_c: Updated constraint values.
      accepted: Whether a step was accepted.
    """
    # Initial values
    alpha = 1.0
    theta_k = jnp.sum(current_c * current_c)  # Constraint violation measure
    phi_k = current_cost
    slope = jnp.sum(q*dX) + jnp.sum(r*dU)

    def continuation_criterion(inputs):
        _, _, _, alpha, accepted = inputs
        return jnp.logical_and(jnp.logical_not(accepted), alpha > alpha_min)

    def body(inputs):
        _, _, _, alpha, _ = inputs

        # Compute trial point
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV

        # Evaluate at new point
        new_cost, new_c = model_evaluator(X_new, U_new)
        theta_new = jnp.sum(new_c * new_c)  # Constraint violation measure
        phi_new = new_cost

        # Case 1: Large constraint violation but improving
        condition1 = theta_new > theta_max
        case1 = jnp.logical_and(
            condition1,
            theta_new < (1 - gamma_theta) * theta_k
        )

        # Case 2: Small constraint violations and cost is decreasing
        condition2 =  jnp.logical_and(
                jnp.maximum(theta_new, theta_k) < theta_min,
                slope < 0
            )
        case2 = jnp.logical_and(
           condition2,
            phi_new < phi_k + eta * alpha * slope
        )

        # Case 3: Either cost or constraint violation is significantly reduced
        condition3 = jnp.logical_not(jnp.logical_or(condition1, condition2))
        case3 = jnp.logical_and(condition3,jnp.logical_or(
            phi_new < phi_k - gamma_phi * phi_k,
            theta_new < (1 - gamma_theta) * theta_k
        ))

        # Accept if any case is satisfied
        new_accepted = jnp.logical_or(jnp.logical_or(case1, case2), case3)

        # If not accepted, reduce alpha
        alpha = jnp.where(new_accepted, alpha, gamma_alpha * alpha)

        return X_new, U_new, V_new, alpha, new_accepted

    # Run the backtracking loop
    X, U, V, alpha, accepted = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, alpha, False)
    )
    
    return X, U, V
@partial(jit, static_argnums=(0))
def parallel_filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_cost,
    current_c,
    q,
    r,
    # Hyperparameters
    alpha_min=1e-4,
    theta_max=1e-2,
    theta_min=1e-6,
    eta=1e-4,
    gamma_phi=1e-6,
    gamma_theta=1e-6,
    gamma_alpha=0.5,
):
    """Performs a backtracking line search.

    Args:
      X_in: [T+1, n] numpy array of current states.
      U_in: [T, m] numpy array of current controls.
      V_in: [T+1, n] numpy array of current multipliers.
      dX: [T+1, n] numpy array of state search direction.
      dU: [T, m] numpy array of control search direction.
      dV: [T+1, n] numpy array of multiplier search direction.
      current_cost: Current cost value (phi_k).
      current_c: Current constraint violation (theta_k).
      alpha_min: Minimum step size.
      theta_max: Maximum acceptable constraint violation.
      theta_min: Minimum constraint violation worth considering.
      eta: Armijo parameter for sufficient decrease.
      gamma_phi: Cost reduction parameter.
      gamma_theta: Constraint violation reduction parameter.
      gamma_alpha: Step size reduction factor.

    Returns:
      X: [T+1, n] numpy array of updated states.
      U: [T, m] numpy array of updated controls.
      V: [T+1, n] numpy array of updated multipliers.
      new_cost: Updated cost value.
      new_c: Updated constraint values.
      accepted: Whether a step was accepted.
    """
    # Initial values
    alpha_values = jnp.exp2(-jnp.arange(11))
    # alpha = 1.0
    theta_k = jnp.sum(current_c * current_c)  # Constraint violation measure
    phi_k = current_cost
    slope = jnp.sum(q*dX) + jnp.sum(r*dU)

    # def continuation_criterion(inputs):
    #     _, _, _, alpha, accepted = inputs
    #     return jnp.logical_and(jnp.logical_not(accepted), alpha > alpha_min)

    def body(alpha):
        # _, _, _, alpha, _ = inputs

        # Compute trial point
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV

        # Evaluate at new point
        new_cost, new_c = model_evaluator(X_new, U_new)
        theta_new = jnp.sum(new_c * new_c)  # Constraint violation measure
        phi_new = new_cost

        # Case 1: Large constraint violation but improving
        condition1 = theta_new > theta_max
        case1 = jnp.logical_and(
            condition1,
            theta_new < (1 - gamma_theta) * theta_k
        )

        # Case 2: Small constraint violations and cost is decreasing
        condition2 =  jnp.logical_and(
                jnp.maximum(theta_new, theta_k) < theta_min,
                slope < 0
            )
        case2 = jnp.logical_and(
           condition2,
            phi_new < phi_k + eta * alpha * slope
        )

        # Case 3: Either cost or constraint violation is significantly reduced
        condition3 = jnp.logical_not(jnp.logical_or(condition1, condition2))
        case3 = jnp.logical_and(condition3,jnp.logical_or(
            phi_new < phi_k - gamma_phi * phi_k,
            theta_new < (1 - gamma_theta) * theta_k
        ))

        # Accept if any case is satisfied
        new_accepted = jnp.logical_or(jnp.logical_or(case1, case2), case3)

        return X_new, U_new, V_new, new_accepted

    # Run the backtracking loop
    X, U, V,accepted = vmap(body)(alpha_values)
    best_index = jnp.where(jnp.any(accepted), jnp.argmax(accepted), -1)

    X_new = X[best_index]
    U_new = U[best_index]
    V_new = V[best_index]
    
    return X_new, U_new, V_new

@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics,x0, X, U):
    """Evaluates the costs and constraints based on the provided primal variables.

    Args:
      cost:            cost function with signature cost(x, u, t).
      dynamics:        dynamics function with signature dynamics(x, u, t).
      x0:              [n]           numpy array.
      X:               [T+1, n]      numpy array.
      U:               [T, m]        numpy array.

    Returns:
      g: the cost value (a scalar).
      c: the constraint values (a [T+1, n] numpy array).
    """
    T = U.shape[0]
    costs = vmap(cost)(X, jnp.pad(U, [[0, 1], [0, 0]]), jnp.arange(T + 1))
    g = jnp.sum(costs)

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = jnp.vstack([x0 - X[0], vmap(residual_fn)(jnp.arange(T))])

    return g, c


@partial(jit, static_argnums=(0,1,2,3,4,5,6,7,8))
def mpc(
    sls_config: SLSConfig,
    sqp_config: SQPConfig,
    admm_config,
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    constraints,
    disturbance,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    w,
    y,
    rho,
    obstacles,
    h_ct_ws,
):
    Tp1 = X_in.shape[0]
    nx = X_in.shape[1]
    nu = U_in.shape[1]
    nc = w.shape[1]
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    _dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics, x0)

    def body(i, carry):
        i, X_curr, U_curr, V_curr, w, y, rho, converged, backoffs, Phi_x, Phi_u = carry

        # If already converged, freeze state (no further work).
        def do_nothing(_):
            return carry

        def do_iter(_):
            # Evaluate current constraints
            g, c = model_evaluator(X_curr, U_curr)

            # Convergence criterion 1: feasibility (max norm of constraint residual)
            feas = jnp.max(jnp.abs(c))

            # Reset inner variables (kept as in your original code)
            warm_flag = jnp.array(bool(sqp_config.warm_start))  # safe because sqp_config is static

            w0   = lax.select(warm_flag, w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            rho0 = lax.select(warm_flag, rho, jnp.asarray(30.0))
            # Compute search direction
            h_ct_ws = backoffs
            dX, dU, dV, q, r, w1, y1, rho1, backoffs1, Phi_x1, Phi_u1 = compute_search_direction(
                sls_config,
                admm_config,
                _cost,
                _dynamics,
                _hessian_approx,
                limited_mempory,
                constraints,
                disturbance,
                obstacles,
                x0,
                X_curr,
                U_curr,
                V_curr,
                c,
                w0, y0, rho0,
                h_ct_ws
            )

            # Convergence criterion 2: relative step size (infinity norm)
            step = jnp.maximum(
                jnp.max(jnp.abs(dX)),
                jnp.max(jnp.abs(dU))
            )
            z_norm = jnp.maximum(
                jnp.max(jnp.abs(X_curr)),
                jnp.max(jnp.abs(U_curr))
            )

            feas_ok = feas <= sqp_config.feas_tol
            step_ok = step <= sqp_config.step_tol * (1.0 + z_norm)
            jax.debug.print("SQP Iteration {} Feas {} (<= {}) Step {} (<= {})", i, feas, sqp_config.feas_tol, step, sqp_config.step_tol)
            converged1 = jnp.logical_and(feas_ok, step_ok)

            # Only apply the step if not converged
            X_next = lax.select(converged1, X_curr, X_curr + dX)
            U_next = lax.select(converged1, U_curr, U_curr + dU)
            V_next = lax.select(converged1, V_curr, V_curr + dV)

            g, c = model_evaluator(X_curr, U_curr)

            rho_merit = merit_rho(c, dV)          # or keep fixed
            merit_fn  = merit_function_factory(rho_merit)
            current_merit = merit_fn(V_curr, g, c)

            merit_slope = slope(dX, dU, dV, c, q, r, rho_merit)

            X_next, U_next, V_next, g_new, c_new, ok = line_search(
                merit_fn, model_evaluator,
                X_curr, U_curr, V_curr,
                dX, dU, dV,
                current_merit, g, c,
                merit_slope,
                armijo_factor=1e-4,
                alpha_0=1.0,
                alpha_mult=0.5,
                alpha_min=1e-6,
            )

            # Keep the latest aux outputs; if converged, keep prior ones
            w_next = lax.select(converged1, w, w1)
            y_next = lax.select(converged1, y, y1)
            rho_next = lax.select(converged1, rho, rho1)
            backoffs_next = lax.select(converged1, backoffs, backoffs1)
            Phi_x_next = lax.select(converged1, Phi_x, Phi_x1)
            Phi_u_next = lax.select(converged1, Phi_u, Phi_u1)

            return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
                    jnp.logical_or(converged, converged1),
                    backoffs_next, Phi_x_next, Phi_u_next)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    # Initialize carry; backoffs/Phi_* placeholders must be valid JAX values
    # If you have natural initial values, use them instead.
    backoffs0 = h_ct_ws
    Phi_x0 = jnp.zeros((Tp1, Tp1, nx, nx))
    Phi_u0 = jnp.zeros((Tp1 - 1, Tp1, nu, nx))

    carry0 = (0, X_in, U_in, V_in, w, y, rho, jnp.array(False), backoffs0, Phi_x0, Phi_u0)
    total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out, converged, backoffs, Phi_x, Phi_u = lax.fori_loop(
        0, sqp_config.max_sqp_iterations, body, carry0
    )
    return X_out, U_out, V_out, w_out, y_out, rho_out, backoffs, Phi_x, Phi_u


@partial(jit, static_argnums=(0,1,2,3,4,5))
def al_mpc(
    cost,
    dynamics,
    eq_constraint,
    ineq_constraint,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    V_equality,
    V_inequality,
    penalty,
    tol
    ):

    # active set

    _eq_constraint = partial(eq_constraint,parameter)
    _ineq_constraint = partial(ineq_constraint,parameter)

    eq_constraint_mapped = vectorize(_eq_constraint)
    ineq_constraint_mapped = vectorize(_ineq_constraint)

    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])
    N = U_in.shape[0]
    # evaluate constraints
    U_pad = pad(U_in)

    equality = eq_constraint_mapped(X_in, U_pad, jnp.arange(N+1))
    inequality = ineq_constraint_mapped(X_in, U_pad, jnp.arange(N+1))


    # active_set = jax.vmap(
    #     lambda t: jnp.where(
    #         jnp.logical_and(jnp.isclose(V_inequality[t], 0.0), jnp.less(inequality[t], 0.0)),
    #         0.0,
    #         1.0
    #     )
    # )(jnp.arange(N+1))

    def augmented_lagrangian(W,reference,x, u, t):

        # stage cost
        J = cost(W,reference,x, u, t)

        # stage equality constraint
        equality = _eq_constraint(x, u, t)

        # stage inequality constraint
        inequality = _ineq_constraint(x, u, t)

        # active_set = jnp.invert(jnp.isclose(V_inequality[t], 0.0) & (inequality < 0.0))
        # active_set = jnp.where(jnp.logical_and(jnp.isclose(V_inequality[t], 0.0),jnp.less(inequality,0.0)), 0.0, 1.0)
        # update cost
        # J += V_equality[t].T @ equality + 0.5 * penalty * equality.T @ equality
        # J += V_inequality[t].T @ inequality + 0.5 * penalty * inequality.T @ (
        #     active_set * inequality )
        # J += 0.5/penalty *(jnp.maximum(inequality + penalty * V_inequality[t],0.0)).T @ (jnp.maximum(inequality + penalty * V_inequality[t],0.0))
        J += 0.5*penalty *(jnp.maximum(inequality,0.0)).T @ (jnp.maximum(inequality,0.0))

        return J

    def augmented_lagrangian_hessian(W,reference,x, u, t):
        # stage cost
        Q, R, M = hessian_approx(W,reference,x, u, t)


        J_eq_x = jax.jacobian(_eq_constraint, argnums=0)
        J_eq_u = jax.jacobian(_eq_constraint, argnums=1)

        J_ineq_x = jax.jacobian(_ineq_constraint, argnums=0)
        J_ineq_u = jax.jacobian(_ineq_constraint, argnums=1)

        # stage inequality constraint
        inequality = _ineq_constraint(x, u, t)

        # active_set = jnp.where(jnp.less(inequality + penalty * V_inequality[t],0.0), 0.0, 1.0)

        active_set = jnp.where(inequality < 0.0, 0.0, 1.0)
        # active_set = jnp.where(jnp.logical_and(jnp.isclose(V_inequality[t], 0.0),jnp.less(inequality,0.0)), 0.0, 1.0)
        penalty_matrix = 0.5*penalty*jnp.diag(active_set)

        Q = Q + J_ineq_x(x, u, t).T @ penalty_matrix @J_ineq_x(x, u, t) #+ 0.5/penalty*J_eq_x(x, u, t).T @ J_eq_x(x, u, t)
        R = R + J_ineq_u(x, u, t).T @ penalty_matrix @J_ineq_u(x, u, t) #+ 0.5/penalty*J_eq_u(x, u, t).T @ J_eq_u(x, u, t)
        M = M + J_ineq_x(x, u, t).T @ penalty_matrix @J_ineq_u(x, u, t) #+ 0.5/penalty*J_eq_x(x, u, t).T @ J_eq_u(x, u, t)

        return Q, R, M

    X, U, V, _ = mpc(
                    augmented_lagrangian,
                    dynamics,
                    augmented_lagrangian_hessian,
                    limited_mempory,
                    reference,
                    parameter,
                    W,
                    x0,
                    X_in,
                    U_in,
                    V_in,
                    )

    def dual_update(constraint, dual, penalty):
        return dual + penalty * constraint

    def inequality_projection(dual):
        return jnp.maximum(dual, 0.0)

    # vectorize

    dual_update_mapped = vmap(dual_update, in_axes=(0, 0, None))
    # evaluate constraints
    U_pad = pad(U)

    equality = eq_constraint_mapped(X, U_pad, jnp.arange(N+1))
    inequality = ineq_constraint_mapped(X, U_pad, jnp.arange(N+1))
    inequality_projected = inequality_projection(inequality)

    # max_constraint_violation = jnp.maximum(
    #     jnp.max(jnp.abs(equality)),
    #     jnp.max(inequality_projected),
    # )

    # max_dynamics_violation_sq = jnp.sum(c * c)

    # augmented Lagrangian update
    V_equality_new = dual_update_mapped(equality, V_equality, penalty)

    V_inequality_new = dual_update_mapped(inequality, V_inequality, penalty)
    V_inequality_new = inequality_projection(V_inequality_new)
     
    penalty *= jnp.where(jnp.max(inequality_projected) > tol, 1.5*penalty, penalty)
    tol *= jnp.where(jnp.max(inequality_projected) > tol, 0.5, 1.0)
    penalty = jnp.minimum(penalty, 1e2)

    X, U, V, V_equality_new, V_inequality_new = jax.lax.cond(
        jnp.max(jnp.abs(equality)) > tol,
        lambda _: (X_in, U_in, V_in, V_equality, V_inequality),
        lambda _: (X, U, V, V_equality_new, V_inequality_new),
        operand=None,
    )

    return X, U, V, V_equality_new, V_inequality_new, penalty, tol