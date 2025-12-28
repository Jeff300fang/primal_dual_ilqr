from jax import debug, grad, jit, lax, scipy, vmap
import jax
import jax.numpy as jnp

from functools import partial

from trajax.optimizers import linearize, quadratize,vectorize
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls_utils import get_etas, get_constraint_tightenings, get_betas, get_controller
from .admm_tvlqr import constrained_solve, ADMMConfig
import time

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

@partial(jit, static_argnums=(0, 1, 2, 3, 4))
def compute_search_direction(
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    constraints, h_ct,
    x0,
    X,
    U,
    V,
    c,
    w, y, rho,
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

    pad = lambda A: jnp.pad(A, ((0, 1), (0, 0)))  # (T,m) -> (T+1,m)
    U_pad = pad(U)
    t = jnp.arange(X.shape[0])  # (T+1,)

    g = vectorize(constraints)(X, U_pad, t)
    f = -g - h_ct

    C, D = linearize(constraints)(X, U_pad, t)

    cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        condense_block_size=5,
        rho_max=50
    )

    # Solve constrained QP for the SQP step (dX, dU)
    dX, dU, dV, w, y, rho, mu, converged = constrained_solve(
        cfg, Q, q, R, r, M, A, B, c, C, D, f, w, y, rho
    )
    def converged_branch(state):
        # state = (w, y, rho)
        return dX, dU, dV, state[0], state[1], state[2], mu, converged

    def not_converged_branch(state):
        w0, y0, rho0 = state
        w_init = jnp.zeros_like(w0)
        y_init = jnp.zeros_like(y0)
        rho_init = jnp.asarray(0.1, dtype=jnp.float32)

        dX2, dU2, dV2, w2, y2, rho2, mu, conv2 = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C, D, f, w_init, y_init, rho_init
        )
        return dX2, dU2, dV2, w2, y2, rho2, mu, conv2

    dX, dU, dV, w, y, rho, mu, converged = lax.cond(
        converged,
        converged_branch,
        not_converged_branch,
        operand=(w, y, rho),
    )

    return dX, dU, dV, q, r, w, y, rho, mu, Q, R, A, B, C, D

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
    """Performs a primal-dual line search on an augmented Lagrangian merit function in parralel fixing the number of steps.

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
    def step_acceptance(merit,alpha):
        return merit > current_merit + alpha * armijo_factor * merit_slope
    alpha_values = jnp.exp2(-jnp.arange(11))
    def body(alpha):
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new) #this cam br probly avoided
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = jnp.where(jnp.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit

    X, U, V, new_g, new_c, new_merit = vmap(body)(alpha_values)
    acceptance = vmap(step_acceptance)(new_merit,alpha_values)
    best_index = jnp.where(jnp.any(acceptance),jnp.argmin(acceptance),0)
    return X[best_index], U[best_index], V[best_index], new_g[best_index], new_c[best_index]

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
@partial(jit, static_argnums=(0,1,2,3,4,5))
def mpc(
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
    ):

    _cost = partial(cost,W,reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None
    _dynamics = partial(dynamics,parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics,x0)
    X_curr = X_in
    U_curr = U_in
    V_curr = V_in
    max_sls_iterations = 1
    px_idx, py_idx = 0, 1
    Tp1 = X_curr.shape[0]
    nc = w.shape[1]
    T = Tp1 - 1
    # TODO: Warm start these?
    beta = jnp.zeros((Tp1, T, nc)) * 1e-10 
    # --------- Fast SLS Loop ---------
    # for i in range(max_sls_iterations):
    #     # Nominal Trajectory Update
    #     g, c = model_evaluator(X_curr, U_curr)
    #     E = disturbance(X_curr[:-1])
    #     h_ct  = get_constraint_tightenings(beta, eps_beta=1e-6)
    #     dX,dU, dV, q, r, w, y, rho, mu, Q, R, A, B, C, D = compute_search_direction(
    #             _cost,
    #             _dynamics,
    #             _hessian_approx,
    #             limited_mempory,
    #             constraints,
    #             h_ct,
    #             x0,
    #             X_curr,
    #             U_curr,
    #             V_curr,
    #             c,
    #             w, y, rho
    #         )
    #     X_curr = X_curr + dX
    #     U_curr = U_curr + dU
    #     V_curr = V_curr + dV
    #     eta = get_etas(mu, beta)
    #     Phi_x, Phi_u = get_controller(Q, R, A, B, C, D, E, eta)
    #     beta = get_betas(C, D, Phi_x, Phi_u)
    #     h_ct  = get_constraint_tightenings(beta, eps_beta=1e-6)
        # jax.debug.print("{}", h_ct)
    # ------- End Fast SLS Loop --------
    # jax.debug.print("p0 = ({}, {})", X_curr[0, px_idx], X_curr[0, py_idx])
    g, c = model_evaluator(X_curr, U_curr)
    h_ct  = get_constraint_tightenings(beta, eps_beta=1e-6)
    dX,dU, dV, q, r, w, y, rho, mu, Q, R, A, B, C, D = compute_search_direction(
            _cost,
            _dynamics,
            _hessian_approx,
            limited_mempory,
            constraints,
            h_ct,
            x0,
            X_curr,
            U_curr,
            V_curr,
            c,
            w, y, rho
        )
    X_curr = X_curr + dX
    U_curr = U_curr + dU
    V_curr = V_curr + dV
    return X_curr, U_curr, V_curr, w, y, rho

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