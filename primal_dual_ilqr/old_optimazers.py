from jax import debug, grad, jit, lax, scipy, vmap
import jax.numpy as np

from functools import partial

from trajax.optimizers import evaluate, linearize, quadratize

from .kkt_helpers import compute_search_direction_kkt, tvlqr_kkt
from .dual_tvlqr import dual_lqr, dual_lqr_backward, dual_lqr_gpu
from .linalg_helpers import (
    invert_symmetric_positive_definite_matrix,
    project_psd_cone,
)
from .primal_tvlqr import tvlqr, tvlqr_gpu, rollout, rollout_gpu


# -----------------------------
# dtype helpers
# -----------------------------
def _dtype_of(*xs, default=np.float32):
    """Pick a reasonable floating dtype from any provided arrays/scalars."""
    for x in xs:
        if hasattr(x, "dtype"):
            dt = x.dtype
            if dt in (np.float16, np.bfloat16, np.float32, np.float64):
                return dt
    return default


def _as_dtype(x, dtype):
    return np.asarray(x, dtype=dtype)


def _fconst(val, dtype):
    """Typed floating constant."""
    return np.asarray(val, dtype=dtype)


# -----------------------------
# Lagrangian + regularization
# -----------------------------
def lagrangian(cost, dynamics, x0):
    """Returns a function to evaluate the associated Lagrangian."""

    def fun(x, u, t, v, v_prev):
        # Force everything involved in arithmetic to a consistent dtype (use x.dtype).
        dt = x.dtype
        x0_ = _as_dtype(x0, dt)
        v_ = _as_dtype(v, dt)
        vprev_ = _as_dtype(v_prev, dt)

        c1 = cost(x, u, t)                     # should already be dt
        c1 = _as_dtype(c1, dt)

        dyn = dynamics(x, u, t)
        dyn = _as_dtype(dyn, dt)
        c2 = np.dot(v_, dyn)

        # IMPORTANT: lax.select does NOT promote; both branches must match dtype.
        sel = lax.select(
            t == 0,
            (x0_ - x).astype(dt),
            (-x).astype(dt),
        )
        c3 = np.dot(vprev_, sel)

        return c1 + c2 + c3

    return fun


@jit
def regularize(Q, R, M, make_psd, psd_delta):
    """Regularizes the Q and R matrices."""
    T, n, m = M.shape
    dt = Q.dtype

    psd = vmap(partial(project_psd_cone, delta=psd_delta))

    # Ensure R is PSD/PD as requested.
    R = lax.cond(make_psd, psd, lambda x: x, R)

    # Ensure Q - M R^{-1} M^T is PSD as requested.
    Rinv = vmap(lambda t: invert_symmetric_positive_definite_matrix(R[t]))(np.arange(T))
    MRinvMT = vmap(lambda t: M[t] @ Rinv[t] @ M[t].T)(np.arange(T))
    QMRinvMT = vmap(lambda t: Q[t] - MRinvMT[t])(np.arange(T))
    QMRinvMT = lax.cond(make_psd, psd, lambda x: x, QMRinvMT)

    Q_T = Q[T].reshape([1, n, n])
    Q_T = lax.cond(make_psd, psd, lambda x: x, Q_T)

    Q = np.concatenate([QMRinvMT + MRinvMT, Q_T]).astype(dt)

    return Q, R


# -----------------------------
# Search direction
# -----------------------------
@partial(jit, static_argnums=(0, 1))
def compute_search_direction(
    cost,
    dynamics,
    x0,
    X,
    U,
    V,
    c,
    make_psd,
    psd_delta,
):
    """Computes the SQP search direction."""
    T = U.shape[0]
    dt = X.dtype

    # Normalize all inputs to X.dtype (critical: avoids float32/float64 mixes).
    x0 = _as_dtype(x0, dt)
    X = _as_dtype(X, dt)
    U = _as_dtype(U, dt)
    V = _as_dtype(V, dt)
    c = _as_dtype(c, dt)

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    # Trajax quadratize/linearize will follow the dtype of the computation graph.
    quadratizer = quadratize(lagrangian(cost, dynamics, x0), argnums=5)
    Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)

    Q = _as_dtype(Q, dt)
    R_pad = _as_dtype(R_pad, dt)
    M_pad = _as_dtype(M_pad, dt)

    R = R_pad[:-1]
    M = M_pad[:-1]

    Q, R = regularize(Q, R, M, make_psd, psd_delta)

    linearizer = linearize(lagrangian(cost, dynamics, x0), argnums=5)
    q, r_pad = linearizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)

    q = _as_dtype(q, dt)
    r_pad = _as_dtype(r_pad, dt)
    r = r_pad[:-1]

    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = _as_dtype(A_pad[:-1], dt)
    B = _as_dtype(B_pad[:-1], dt)

    # GPU tvlqr/rollout
    K, k, P, p = tvlqr_gpu(Q, q, R, r, M, A, B, c[1:])
    dX, dU = rollout_gpu(K, k, c[0], A, B, c[1:])
    dV = dual_lqr(dX, P, p)

    # Normalize outputs
    dX = _as_dtype(dX, dt)
    dU = _as_dtype(dU, dt)
    dV = _as_dtype(dV, dt)

    return dX, dU, dV, q, r


# -----------------------------
# Merit function components
# -----------------------------
@jit
def merit_rho(c, dV):
    """Determines the merit function penalty parameter to be used."""
    dt = c.dtype
    c2 = np.sum(c * c).astype(dt)
    dV2 = np.sum(dV * dV).astype(dt)

    eps = _fconst(1e-12, dt)
    two = _fconst(2.0, dt)
    small = _fconst(1e-2, dt)

    # Use where/select with typed constants; keep dtype stable.
    return np.where(c2 > eps, two * np.sqrt(dV2 / c2), small).astype(dt)


@jit
def slope(dX, dU, dV, c, q, r, rho):
    """Directional derivative of the merit function."""
    dt = dX.dtype
    rho = _as_dtype(rho, dt)
    return (
        np.sum(q * dX)
        + np.sum(r * dU)
        + np.sum(dV * c)
        - rho * np.sum(c * c)
    ).astype(dt)


# -----------------------------
# Line search
# -----------------------------
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
    """Performs a primal-dual line search on an augmented Lagrangian merit function."""
    dt = X_in.dtype

    # Typed scalars (alpha-related comparisons in lax.while_loop are dtype-sensitive).
    armijo_factor = _fconst(armijo_factor, dt)
    alpha_0 = _fconst(alpha_0, dt)
    alpha_mult = _fconst(alpha_mult, dt)
    alpha_min = _fconst(alpha_min, dt)

    current_merit = _as_dtype(current_merit, dt)
    merit_slope = _as_dtype(merit_slope, dt)

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        return np.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha = (alpha * alpha_mult).astype(dt)

        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV

        new_g, new_c = model_evaluator(X_new, U_new)
        new_g = _as_dtype(new_g, dt)
        new_c = _as_dtype(new_c, dt)

        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = _as_dtype(new_merit, dt)

        new_merit = np.where(np.isnan(new_merit), current_merit, new_merit).astype(dt)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    inf = _fconst(np.inf, dt)

    X, U, V, new_g, new_c, _, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, inf, alpha_0 / alpha_mult),
    )

    no_errors = alpha > alpha_min

    return X, U, V, new_g, new_c, no_errors


# -----------------------------
# Model evaluator
# -----------------------------
@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics, x0, X, U):
    """Evaluates the costs and constraints based on the provided primal variables."""
    dt = X.dtype
    x0 = _as_dtype(x0, dt)
    X = _as_dtype(X, dt)
    U = _as_dtype(U, dt)

    T = U.shape[0]

    costs = partial(evaluate, cost)
    g = np.sum(costs(X, np.pad(U, [[0, 1], [0, 0]])))
    g = _as_dtype(g, dt)

    def residual_fn(t):
        dyn = dynamics(X[t], U[t], t)
        dyn = _as_dtype(dyn, dt)
        return dyn - X[t + 1]

    c = np.vstack([x0 - X[0], vmap(residual_fn)(np.arange(T))])
    c = _as_dtype(c, dt)

    return g, c


# -----------------------------
# Main primal-dual iLQR
# -----------------------------
@partial(jit, static_argnums=(0, 1))
def primal_dual_ilqr(
    cost,
    dynamics,
    x0,
    X_in,
    U_in,
    V_in,
    max_iterations=100,
    slope_threshold=1e-4,
    var_threshold=0.0,
    c_sq_threshold=1e-4,
    make_psd=True,
    psd_delta=1e-6,
    armijo_factor=1e-4,
    alpha_0=1.0,
    alpha_mult=0.5,
    alpha_min=5e-5,
):
    """Implements the Primal-Dual iLQR algorithm."""

    # Choose a single dtype and normalize everything to it.
    dt = _dtype_of(X_in, U_in, V_in, x0, default=np.float32)
    X_in = _as_dtype(X_in, dt)
    U_in = _as_dtype(U_in, dt)
    V_in = _as_dtype(V_in, dt)
    x0 = _as_dtype(x0, dt)

    # Typed thresholds/constants used inside while_loop conditions.
    slope_threshold = _fconst(slope_threshold, dt)
    var_threshold = _fconst(var_threshold, dt)
    c_sq_threshold = _fconst(c_sq_threshold, dt)

    model_evaluator = partial(model_evaluator_helper, cost, dynamics, x0)

    @jit
    def merit_function(V, g, c, rho):
        # Ensure rho is typed and broadcast-safe
        rho = _as_dtype(rho, c.dtype)
        return (g + np.sum((V + _fconst(0.5, c.dtype) * rho * c) * c)).astype(c.dtype)

    @jit
    def direction_and_merit(X, U, V, g, c):
        # If you want KKT direction instead, keep it but still dtype-normalize there.
        dX, dU, dV, q, r = compute_search_direction(
            cost,
            dynamics,
            x0,
            X,
            U,
            V,
            c,
            make_psd,
            psd_delta,
        )

        rho = merit_rho(c, dV)
        merit = merit_function(V, g, c, rho)

        merit_slope = slope(dX, dU, dV, c, q, r, rho)

        return dX, dU, dV, rho, merit, merit_slope

    def body(inputs):
        """Solves LQR subproblem and returns updated trajectory."""
        X, U, V, dX, dU, dV, iteration, _, g, c, rho, merit, merit_slope = inputs

        X_new, U_new, V_new, g_new, c_new, no_errors = line_search(
            partial(merit_function, rho=rho),
            model_evaluator,
            X,
            U,
            V,
            dX,
            dU,
            dV,
            merit,
            g,
            c,
            merit_slope,
            armijo_factor,
            alpha_0,
            alpha_mult,
            alpha_min,
        )

        dX_new, dU_new, dV_new, rho_new, merit_new, merit_slope_new = direction_and_merit(
            X_new, U_new, V_new, g_new, c_new
        )

        return (
            X_new,
            U_new,
            V_new,
            dX_new,
            dU_new,
            dV_new,
            iteration + 1,
            no_errors,
            g_new,
            c_new,
            rho_new,
            merit_new,
            merit_slope_new,
        )

    def continuation_criterion(inputs):
        _, _, _, dX, dU, dV, iteration, no_errors, _, c, _, _, slope_val = inputs

        c_sq_norm = np.sum(c * c).astype(dt)
        slope_ok = np.abs(slope_val) > slope_threshold
        delta_norm_sq = (np.sum(dX * dX) + np.sum(dU * dU) + np.sum(dV * dV)).astype(dt)
        delta_norm_ok = delta_norm_sq > var_threshold * var_threshold
        c_ok = c_sq_norm > c_sq_threshold

        progress_ok = np.logical_or(np.logical_and(slope_ok, delta_norm_ok), c_ok)
        status_ok = np.logical_and(no_errors, iteration < max_iterations)

        return np.logical_and(status_ok, progress_ok)

    g, c = model_evaluator(X_in, U_in)
    g = _as_dtype(g, dt)
    c = _as_dtype(c, dt)

    dX, dU, dV, rho, merit, merit_slope = direction_and_merit(X_in, U_in, V_in, g, c)

    X, U, V, _, _, _, iteration, no_errors, g, c, _, _, merit_slope = lax.while_loop(
        continuation_criterion,
        body,
        (
            X_in,
            U_in,
            V_in,
            dX,
            dU,
            dV,
            0,
            True,
            g,
            c,
            rho,
            merit,
            merit_slope,
        ),
    )

    no_errors = np.logical_and(no_errors, iteration < max_iterations)

    return X, U, V, iteration, g, c, no_errors
