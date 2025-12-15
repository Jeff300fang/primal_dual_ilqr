from functools import partial
import time
import jax
import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass
from jax import lax
import math
import jax.scipy.linalg as jsp
from jax import profiler, config
from typing import Tuple

# jax.config.update('jax_enable_x64', True)
# config.update("jax_default_matmul_precision", "tensorfloat32")

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass
class ADMMProblem:
    Q: Array
    q: Array
    R: Array
    r: Array
    A: Array
    B: Array
    c: Array
    C: Array
    D: Array
    f: Array
    N: int
    nx: int
    nu: int
    nc: int

    def tree_flatten(self):
        children = (self.Q, self.q, self.R, self.r,
                    self.A, self.B, self.c, self.C, self.D, self.f)
        aux = (self.N, self.nx, self.nu, self.nc)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        N, nx, nu, nc = aux
        Q, q, R, r, A, B, c, C, D, f = children
        return cls(Q, q, R, r, A, B, c, C, D, f, N, nx, nu, nc)

@dataclass
class ADMMConfig:
    rho_update_frequency: int = 25
    max_iterations: int = 400
    eps_abs: float = 1e-2
    eps_rel: float = 1e-2
    condense_block_size: int = 1


class ADMMPCR:
    def __init__(self,
                 admm_config: ADMMConfig):
        """ADMM-based QP solver.

        Note: the problem data is passed into `solve(problem, x0, ...)` so that
        the solver can be used with jitted call-sites without capturing large
        problem tensors in the Python object.
        """
        self.rho_update_frequency = admm_config.rho_update_frequency
        self.max_iterations = admm_config.max_iterations
        self.condense_block_size = admm_config.condense_block_size
        self.eps_abs = admm_config.eps_abs
        self.eps_rel = admm_config.eps_rel

        # Warm-start buffers (condensed-space ADMM state).
        self._z_ws = None
        self._w_ws = None
        self._y_ws = None
        self._rho_ws = None

    # ------- Linalg Heleprs -------
    def cholesky_factor(self, A, eps: float = 1e-9):
        """
        JAX-friendly 'robust' Cholesky factorization.

        - Symmetrizes A.
        - Projects to SPD via eigenvalue clipping.
        - Returns Cholesky factor of the projected SPD matrix.

        This is fully jit-compatible (no Python exceptions, no np).
        """
        # Symmetrize
        A_sym = 0.5 * (A + A.T)

        # Useful for non SPD
        # # Eigen-decompose
        # w, V = jnp.linalg.eigh(A_sym)  # w: (n,), V: (n,n)

        # # Clip eigenvalues to ensure SPD
        # w_clipped = jnp.clip(w, eps, None)

        # Reconstruct SPD matrix: V diag(w_clipped) V^T
        # (V * w_clipped) broadcasts w_clipped over columns of V
        # A_spd = (V * w_clipped) @ V.T

        A_spd = A_sym + eps * jnp.eye(A.shape[0], dtype=A.dtype)
        # Cholesky factor
        L = jnp.linalg.cholesky(A_spd)
        return L

    def spd_solve_from_factor(self, L, B):
        B2 = B if B.ndim == 2 else B[:, None]  # (n,k)
        X2 = jsp.cho_solve((L, True), B2)
        return X2 if B.ndim == 2 else X2[:, 0]

    def spd_solve(self, A, B, jitter=1e-6):
        """
        JAX-friendly SPD solve:
            A X = B

        Uses:
        - symmetrization
        - diagonal jitter for safety
        - Cholesky factorization
        """
        # Symmetrize
        A_sym = 0.5 * (A + A.T)

        # Add jitter to ensure positivity
        A_spd = A_sym + jitter * jnp.eye(A.shape[0], dtype=A.dtype)

        # Cholesky factorization
        L = jnp.linalg.cholesky(A_spd)

        return jsp.cho_solve((L, True), B)

    # ------- Partial C + Full Condensing -------
    def condense_problem_partial(self,
                             Q, q, R, r, A, B, c, C, D, f,
                             N, nx, nu, nc, m):
        """
        Parallel (over blocks) JAX version of partial condensing with block size m.

        Horizon: 0..N
        - N_blocks = N // m  (assume N % m == 0)
        - Each block j covers stages k = j*m .. j*m + (m-1)

        We build quadratic blocks Hs[j] over v_j = [x_j; u_bar_j],
        where u_bar_j = [u_{jm}, ..., u_{jm+m-1}] stacked.
        """

        N_blocks = N // m
        nu_block = m * nu
        nc_block = m * nc
        nv_block = nx + nu_block

        I_nx = jnp.eye(nx)

        # --------- Reshape horizon data into (N_blocks, m, ...) ---------
        # We assume N = N_blocks * m
        Q_blocks = Q[:N_blocks * m].reshape(N_blocks, m, nx, nx)
        q_blocks = q[:N_blocks * m].reshape(N_blocks, m, nx)
        R_blocks = R[:N_blocks * m].reshape(N_blocks, m, nu, nu)
        r_blocks = r[:N_blocks * m].reshape(N_blocks, m, nu)
        A_blocks = A[:N_blocks * m].reshape(N_blocks, m, nx, nx)
        B_blocks = B[:N_blocks * m].reshape(N_blocks, m, nx, nu)
        c_blocks = c[:N_blocks * m].reshape(N_blocks, m, nx)
        C_blocks = C[:N_blocks * m].reshape(N_blocks, m, nc, nx)
        D_blocks = D[:N_blocks * m].reshape(N_blocks, m, nc, nu)
        f_blocks = f[:N_blocks * m].reshape(N_blocks, m, nc, 1)

        # --------- Per-block condensing (sequential in m, parallel over blocks) ---------
        def condense_one_block(Q_blk, q_blk, R_blk, r_blk,
                       A_blk, B_blk, c_blk, C_blk, D_blk, f_blk):
            """
            Affine dynamics inside block:
                x_{ell+1} = A_ell x_ell + B_ell u_ell + c_ell
            So within the block we represent:
                x_ell = Phi[ell] x0_block + sum_tau Psi[ell,tau] u_tau + s[ell]
            where s[0]=0 and s[ell+1] = A_ell s[ell] + c_ell.
            """
            nx = Q_blk.shape[1]
            nu = R_blk.shape[2]
            nc = C_blk.shape[1]
            m  = Q_blk.shape[0]

            nv_block = nx + m * nu
            nc_block = m * nc

            I_nx = jnp.eye(nx)

            # --- Phi, Psi, and affine offset s ---
            Phi = jnp.zeros((m + 1, nx, nx))
            Phi = Phi.at[0].set(I_nx)
            Psi = jnp.zeros((m + 1, m, nx, nu))
            s   = jnp.zeros((m + 1, nx))  # affine offsets

            def ell_body(ell, PhiPsiS):
                Phi, Psi, s = PhiPsiS
                A_k = A_blk[ell]
                B_k = B_blk[ell]
                c_k = c_blk[ell]

                # Phi
                Phi_next = A_k @ Phi[ell]
                Phi = Phi.at[ell + 1].set(Phi_next)

                # Psi propagation
                def psi_tau_body(tau, Psi_inner):
                    prev = Psi_inner[ell, tau]
                    new  = A_k @ prev
                    Psi_inner = Psi_inner.at[ell + 1, tau].set(new)
                    return Psi_inner

                Psi = lax.fori_loop(0, ell, psi_tau_body, Psi)
                Psi = Psi.at[ell + 1, ell].set(B_k)

                # affine offset
                s_next = A_k @ s[ell] + c_k
                s = s.at[ell + 1].set(s_next)

                return (Phi, Psi, s)

            Phi, Psi, s = lax.fori_loop(0, m, ell_body, (Phi, Psi, s))

            # --- Initialize accumulators ---
            H_j = jnp.zeros((nv_block, nv_block))
            h_j = jnp.zeros((nv_block,))
            C_j = jnp.zeros((nc_block, nv_block))
            f_j = jnp.zeros((nc_block, 1))

            # helper funcs unchanged
            def set_cols(mat, col_start, block):
                return lax.dynamic_update_slice(mat, block, (0, col_start))

            def add_cols(mat, col_start, block):
                rows = mat.shape[0]
                nu_  = block.shape[1]
                cur  = lax.dynamic_slice(mat, (0, col_start), (rows, nu_))
                new  = cur + block
                return lax.dynamic_update_slice(mat, new, (0, col_start))

            def set_rows(mat, row_start, block):
                return lax.dynamic_update_slice(mat, block, (row_start, 0))

            def set_rows_vec(mat, row_start, block):
                return lax.dynamic_update_slice(mat, block, (row_start, 0))

            # --- Accumulate over stages in the block ---
            def cost_body(ell, carry):
                H_j, h_j, C_j, f_j = carry

                Q_k = Q_blk[ell]
                q_k = q_blk[ell]
                R_k = R_blk[ell]
                r_k = r_blk[ell]
                C_k = C_blk[ell]
                D_k = D_blk[ell]
                f_k = f_blk[ell]

                Phi_l = Phi[ell]
                s_l   = s[ell]  # (nx,)

                # Build U_l
                U_l = jnp.zeros((nx, m * nu))

                def fill_U_body(tau, U_inner):
                    col = tau * nu
                    upd = Psi[ell, tau]
                    return set_cols(U_inner, col, upd)

                U_l = lax.fori_loop(0, ell, fill_U_body, U_l)

                # M_l = [Phi_l  U_l]
                M_l = jnp.zeros((nx, nv_block))
                M_l = lax.dynamic_update_slice(M_l, Phi_l, (0, 0))
                M_l = lax.dynamic_update_slice(M_l, U_l,   (0, nx))

                # ----- State cost with affine offset -----
                # x = M_l v + s_l
                # 0.5 x^T Q x + q^T x
                # => quad: 0.5 v^T (M^T Q M) v
                #    lin:  v^T M^T (Q s + q)
                H_j = H_j + M_l.T @ Q_k @ M_l
                h_j = h_j + (M_l.T @ (Q_k @ s_l + q_k))

                # ----- Input cost for u_ell -----
                u_start = nx + ell * nu
                H_block = lax.dynamic_slice(H_j, (u_start, u_start), (nu, nu))
                H_block = H_block + R_k
                H_j = lax.dynamic_update_slice(H_j, H_block, (u_start, u_start))

                h_block = lax.dynamic_slice(h_j, (u_start,), (nu,))
                h_block = h_block + r_k
                h_j = lax.dynamic_update_slice(h_j, h_block, (u_start,))

                # ----- Constraints with affine offset -----
                # C_k x + D_k u <= f_k
                # => C_k (M_l v + s_l) + D_u v <= f_k
                # => [C_k M_l, D_u] v <= f_k - C_k s_l
                C_x = C_k @ Phi_l
                D_u = jnp.zeros((nc, m * nu))

                def fill_D_body(tau, D_inner):
                    col = tau * nu
                    contrib = C_k @ Psi[ell, tau]  # (nc,nu)
                    rows = nc
                    nu_loc = contrib.shape[1]
                    cur = lax.dynamic_slice(D_inner, (0, col), (rows, nu_loc))
                    upd = cur + contrib
                    return lax.dynamic_update_slice(D_inner, upd, (0, col))

                D_u = lax.fori_loop(0, ell, fill_D_body, D_u)
                D_u = add_cols(D_u, ell * nu, D_k)

                row_start = ell * nc
                C_row = jnp.concatenate([C_x, D_u], axis=1)

                C_j = set_rows(C_j, row_start, C_row)
                f_j = set_rows_vec(f_j, row_start, f_k - (C_k @ s_l).reshape(nc, 1))

                return (H_j, h_j, C_j, f_j)

            H_j, h_j, C_j, f_j = lax.fori_loop(0, m, cost_body, (H_j, h_j, C_j, f_j))
            H_j = 0.5 * (H_j + H_j.T)

            # --- Macro dynamics: x_{j+1} = Phi[m] x_j + B_bar u_bar + d_bar ---
            Phi_m = Phi[m]
            B_bar = jnp.zeros((nx, m * nu))

            def B_body(tau, B_inner):
                col = tau * nu
                upd = Psi[m, tau]
                return set_cols(B_inner, col, upd)

            B_bar = lax.fori_loop(0, m, B_body, B_bar)
            A_j = jnp.concatenate([Phi_m, B_bar], axis=1)

            d_bar = s[m]  # (nx,)

            return H_j, h_j, A_j, d_bar, C_j, f_j

        # vmap over the block axis: (N_blocks, ...) -> (N_blocks, ...)
        H_blocks, h_blocks, A_blocks_out, d_blocks_out, C_blocks, f_blocks_out = jax.vmap(
            condense_one_block, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )(Q_blocks, q_blocks, R_blocks, r_blocks,
        A_blocks, B_blocks, c_blocks, C_blocks, D_blocks, f_blocks)

        # ---------- Terminal stage at x_N ----------
        QN = Q[N]
        qN = q[N]
        C_N = C[N]
        f_N = f[N]

        M_term = jnp.zeros((nx, nv_block))
        M_term = M_term.at[:, :nx].set(I_nx)

        HN = M_term.T @ QN @ M_term
        hN = M_term.T @ qN
        HN = 0.5 * (HN + HN.T)

        # Assemble final (N_blocks+1, ...) outputs
        Hs = jnp.concatenate([H_blocks, HN[jnp.newaxis, ...]], axis=0)
        hs = jnp.concatenate([h_blocks, hN[jnp.newaxis, ...]], axis=0)

        # Cs_bar and f_bar: last block has only the terminal constraint rows filled at top
        Cs_bar = jnp.zeros((N_blocks + 1, nc_block, nv_block))
        f_bar = jnp.zeros((N_blocks + 1, nc_block, 1))

        Cs_bar = Cs_bar.at[:N_blocks].set(C_blocks)
        f_bar = f_bar.at[:N_blocks].set(f_blocks_out)

        C_N_block = jnp.concatenate(
            [C_N, jnp.zeros((nc, nu_block))],
            axis=1,
        )
        Cs_bar = Cs_bar.at[N_blocks, :nc, :].set(C_N_block)
        f_bar = f_bar.at[N_blocks, :nc, :].set(f_N)

        As_bar = A_blocks_out  # (N_blocks, nx, nv_block)
        d_bar = jnp.concatenate([d_blocks_out, jnp.zeros((1, nx), dtype=d_blocks_out.dtype)], axis=0)

        return Hs, hs, As_bar, d_bar, Cs_bar, f_bar, N_blocks, nv_block, nc_block

    def unpartially_condense(self,
                         m: int,
                         z_plus: Array,
                         nu_orig: int,
                         nx: int,
                         N_eff: int,
                         x0: Array,
                         A_orig: Array,
                         B_orig: Array,
                         c_orig: Array):
        N_blocks = N_eff
        N_orig   = N_eff * m

        z_blocks = z_plus[:N_blocks]
        u_stacked = z_blocks[:, nx:]
        u_full = u_stacked.reshape(N_orig, nu_orig)
        c_use = jnp.reshape(c_orig[:N_orig], (N_orig, nx))  

        def step(x_k, inputs):
            A_k, B_k, c_k, u_k = inputs
            x_next = A_k @ x_k + B_k @ u_k + c_k
            return x_next, x_next

        _, xs = lax.scan(step, x0, (A_orig, B_orig, c_use, u_full))
        x_full = jnp.vstack([x0[None, :], xs])

        return x_full, u_full


    # Logn, but in reality not much difference
    # def unpartially_condense(self,
    #                      m: int,
    #                      z_plus: Array,
    #                      nu_orig: int,
    #                      nx: int,
    #                      N_eff: int,
    #                      x0: Array,
    #                      A_orig: Array,
    #                      B_orig: Array,
    #                      c_orig: Array):
    #     """
    #     Reconstructs (x,u) on the original grid using a parallel associative scan.

    #     Uses affine map composition:
    #         x_{k+1} = A_k x_k + B_k u_k + c_k  :=  M_k x_k + b_k
    #     and composes (M,b) pairs with an associative operator.
    #     """
    #     N_blocks = N_eff
    #     N_orig   = N_eff * m

    #     # ---- Recover original controls u_full from condensed z ----
    #     z_blocks  = z_plus[:N_blocks]                # (N_eff, nv_eff)
    #     u_stacked = z_blocks[:, nx:]                 # (N_eff, m*nu_orig)
    #     u_full    = u_stacked.reshape(N_orig, nu_orig)

    #     # ---- Build per-step affine terms (M_k, b_k) ----
    #     # A_seq = A_orig[:N_orig]                      # (N_orig, nx, nx)
    #     # B_seq = B_orig[:N_orig]                      # (N_orig, nx, nu_orig)
    #     A_seq = A_orig
    #     B_seq = B_orig
    #     c_seq = jnp.reshape(c_orig[:N_orig], (N_orig, nx))  # (N_orig, nx)
    #     # b_k = B_k u_k + c_k
    #     b_seq = jnp.einsum("kij,kj->ki", B_seq, u_full) + c_seq  # (N_orig, nx)

    #     # ---- Associative composition: (M2,b2) ∘ (M1,b1) ----
    #     def compose(left, right):
    #         M1, b1 = left
    #         M2, b2 = right
    #         M = jnp.matmul(M2, M1)
    #         b = jnp.matmul(M2, b1[..., None])[..., 0] + b2
    #         return (M, b)


    #     # Prefix compositions: pref[k] = T_k ∘ ... ∘ T_0
    #     M_pref, b_pref = lax.associative_scan(compose, (A_seq, b_seq), axis=0)

    #     # x_{k+1} = M_pref[k] x0 + b_pref[k]
    #     xs = jnp.einsum("kij,j->ki", M_pref, x0) + b_pref  # (N_orig, nx)

    #     x_full = jnp.vstack([x0[None, :], xs])  # (N_orig+1, nx)
    #     return x_full, u_full


    # ------- Matrix Precalculations / Caching

    def assemble_F(self, Hs, Cs, rho, N):
        """
        Hs: (N+1, nv, nv)
        Cs: (N+1, nc, nv)
        rho: scalar
        N:   horizon (we just assume Hs.shape[0] == N+1)

        Returns:
            F: (N+1, nv, nv), where F[k] = Hs[k] + rho * Cs[k]^T Cs[k]
        """

        # Optionally slice to first N+1 in case arrays are bigger
        Hs_use = Hs[:N+1]
        Cs_use = Cs[:N+1]

        def per_stage(Hk, Ck):
            # Ck: (nc, nv) → Ck.T @ Ck: (nv, nv)
            return Hk + rho * (Ck.T @ Ck)

        F = jax.vmap(per_stage, in_axes=(0, 0))(Hs_use, Cs_use)
        return F

    def assemble_first_schur(self, F, G, H, N, nx, nu):
        """
        JAX / vmap version of the first Schur layer assembly.

        Inputs:
            F : (N+1, nv, nv)
            G : (N,   nx, nv)
            H : (nv,  nx)
        where nv = nx + nu.

        Returns:
            GF_inv  : (N,   nx, nv)
            HTF_inv : (N+1, nx, nv)
            F_chol  : (N+1, nv, nv)
        """
        nv = nx + nu

        # ---------- Cholesky factors ----------
        # Special F0: enforce x0 block as identity, no coupling
        F0 = F[0]
        F0_mod = F0
        F0_mod = F0_mod.at[:nx, :].set(0.0)
        F0_mod = F0_mod.at[:, :nx].set(0.0)
        F0_mod = F0_mod.at[:nx, :nx].set(jnp.eye(nx))

        chol0 = self.cholesky_factor(F0_mod)       # (nv, nv)

        # Factors for k = 1..N
        F_rest = F[1:N+1]                          # (N, nv, nv)
        chol_rest = jax.vmap(self.cholesky_factor, in_axes=(0,))(F_rest)

        F_chol = jnp.concatenate(
            [chol0[jnp.newaxis, ...], chol_rest],
            axis=0,
        )  # (N+1, nv, nv)

        # ---------- GF_inv ----------
        # GF_inv[0] = solve(F_chol[0], G[0].T).T
        GF0 = self.spd_solve_from_factor(F_chol[0], G[0].T).T  # (nx, nv)

        # GF_inv[k] for k = 1..N-1
        def solve_GF(Fk, Gk):
            return self.spd_solve_from_factor(Fk, Gk.T).T   # (nx, nv)

        if N > 1:
            GF_mid = jax.vmap(solve_GF, in_axes=(0, 0))(
                F_chol[1:N],     # (N-1, nv, nv)
                G[1:N],          # (N-1, nx, nv)
            )                    # (N-1, nx, nv)
        else:
            GF_mid = jnp.zeros((0, nx, nv))

        GF_inv = jnp.zeros((N, nx, nv))
        GF_inv = GF_inv.at[0].set(GF0)
        GF_inv = GF_inv.at[1:].set(GF_mid)

        # ---------- HTF_inv ----------
        # HTF_inv[0] stays zero
        # HTF_inv[k] for k = 1..N: solve(F_chol[k], H).T
        def solve_HT(Fk):
            return self.spd_solve_from_factor(Fk, H).T      # (nx, nv)

        HT_mid = jax.vmap(solve_HT, in_axes=(0,))(
            F_chol[1:N+1]    # (N, nv, nv), indices 1..N
        )                    # (N, nx, nv)

        HTF_inv = jnp.zeros((N + 1, nx, nv))
        HTF_inv = HTF_inv.at[1:].set(HT_mid)

        return GF_inv, HTF_inv, F_chol

    def condense_cache(self, F, GF_inv, HTF_inv, G, H, N, nx):
        """
        JAX version of the first-level Schur block (S, V) assembly.

        Inputs:
            F       : (N+1, nv, nv)       # not actually used here, kept for API parity
            GF_inv  : (N,   nx, nv)
            HTF_inv : (N+1, nx, nv)
            G       : (N,   nx, nv)
            H       : (nv,  nx)
            N       : horizon blocks
            nx      : state dimension

        Returns:
            S : (N,   nx, nx)
            V : (N-1, nx, nx)
        """
        # Transpose G per stage: (N, nv, nx)
        G_T = jnp.swapaxes(G, -1, -2)

        # ---------- Base S for all k = 0..N-1 ----------
        # term1[k] = GF_inv[k] @ G[k].T
        term1 = jnp.matmul(GF_inv, G_T)  # (N, nx, nx)

        # term2[k] = HTF_inv[k+1] @ H  for k=0..N-1
        HT_slice = HTF_inv[1:N+1]       # (N, nx, nv)
        term2 = jnp.matmul(HT_slice, H) # (N, nx, nx)

        S = term1 + term2               # (N, nx, nx)

        # ---------- Fix S[0] with initial-conditioning G0T modification ----------
        # Original code zeroed out the top-left (nx x nx) block of G[0].T before using it.
        G0T_mod = G_T[0].at[:nx, :nx].set(jnp.zeros((nx, nx)))
        S0 = GF_inv[0] @ G0T_mod + HTF_inv[1] @ H
        S = S.at[0].set(S0)

        # ---------- V blocks: k = 0..N-2 ----------
        # V[k] = HTF_inv[k+1] @ G[k+1].T
        HT_mid   = HTF_inv[1:N]                    # (N-1, nx, nv)
        G_mid_T  = jnp.swapaxes(G[1:N], -1, -2)    # (N-1, nv, nx)
        V = jnp.matmul(HT_mid, G_mid_T)            # (N-1, nx, nx)

        return S, V

    # ------- PCR -------
    def reduce_to_single_system_caches(self, S_blk, V_lo):
        """
        JAX + batched version of forward reduction caches for odd-even cyclic reduction.

        Inputs:
            S_blk : (N,   n, n)    diagonal blocks
            V_lo  : (N-1, n, n)    lower off-diagonal blocks

        Returns:
            S_levels          : (L_max+1, N,   n, n)
            V_levels          : (L_max+1, N-1, n, n)
            VTS_inv           : (L_max+1, N,   n, n)
            V_S_inv1          : (L_max+1, N,   n, n)
            S_factors_levels  : (L_max+1, N,   n, n)
        """
        N = S_blk.shape[0]
        n = S_blk.shape[1]

        # ---------- Max number of levels ----------
        L_max = int(math.ceil(math.log2(N)))
        if (N & (N - 1)) == 0:
            L_max += 1

        # Preallocate level buffers (JAX arrays)
        S_levels = jnp.zeros((L_max + 1, N,   n, n))
        V_levels = jnp.zeros((L_max + 1, N-1, n, n))

        VTS_inv          = jnp.zeros((L_max + 1, N,   n, n))
        V_S_inv1         = jnp.zeros((L_max + 1, N,   n, n))
        S_factors_levels = jnp.zeros((L_max + 1, N,   n, n))

        # Keep N_levels as Python ints (static)
        N_levels = [0] * (L_max + 1)

        # Level 0 (original system)
        S_levels = S_levels.at[0, :N].set(S_blk)
        V_levels = V_levels.at[0, :N-1].set(V_lo)
        N_levels[0] = N

        # ---------- Forward reduction over levels ----------
        for lev in range(L_max - 1):
            N_cur = N_levels[lev]
            N_next = max(1, N_cur // 2)
            N_levels[lev + 1] = N_next

            S_cur = S_levels[lev]    # (N,   n, n)
            V_cur = V_levels[lev]    # (N-1, n, n)

            # ------------------------------------------------
            # 1) Cache Cholesky factors for neighbors of odd nodes
            #    (parallel over j via vmap)
            # ------------------------------------------------
            # odd positions: i_pos = 2*j + 1, j = 0..N_next-1
            j_idx = jnp.arange(N_next)
            i_pos = 2 * j_idx + 1

            # left neighbors: i_left = i_pos - 1 (even indices)
            i_left = i_pos - 1                    # shape (N_next,)
            S_left = S_cur[i_left]                # (N_next, n, n)

            # batched cholesky on left neighbors
            L_left = jax.vmap(self.cholesky_factor, in_axes=(0,))(S_left)  # (N_next, n, n)
            S_factors_levels = S_factors_levels.at[lev, i_left].set(L_left)

            # right neighbor for the *last* odd node (if exists)
            j_last = N_next - 1
            i_pos_last = int(2 * j_last + 1)
            if (i_pos_last + 1) < N_cur:
                i_right_last = i_pos_last + 1
                L_right_last = self.cholesky_factor(S_cur[i_right_last])
                S_factors_levels = S_factors_levels.at[lev, i_right_last].set(L_right_last)

            # ------------------------------------------------
            # 2) Build S_next[0..N_next-2] and caches VTS_inv, V_S_inv1 for interior odds
            #    (parallel over j = 0..N_next-2 via vmap)
            # ------------------------------------------------
            if N_next > 1:
                j_int = jnp.arange(N_next - 1)
                i_pos_int   = 2 * j_int + 1           # odd indices
                i_left_int  = i_pos_int - 1          # their left neighbors
                i_right_int = i_pos_int + 1          # their right neighbors (exist for interior)

                Sii_int   = S_cur[i_pos_int]         # (N_next-1, n, n)
                L_left_int  = S_factors_levels[lev, i_left_int]   # (N_next-1, n, n)
                L_right_int = S_factors_levels[lev, i_right_int]  # (N_next-1, n, n)

                V_left_int = V_cur[i_left_int]       # (N_next-1, n, n)
                V_mid_int  = V_cur[i_pos_int]        # (N_next-1, n, n)

                # batched solves for VTS_inv and V_S_inv1
                # VTS_inv = solve(S_{i_left}, V_left)^T
                Y_left = jax.vmap(self.spd_solve_from_factor, in_axes=(0, 0))(L_left_int, V_left_int)
                VTS_int = jnp.swapaxes(Y_left, -1, -2)   # transpose each (n,n)

                # V_S_inv1 = solve(S_{i_right}, V_mid^T)^T
                V_mid_T_int = jnp.swapaxes(V_mid_int, -1, -2)
                Y_right = jax.vmap(self.spd_solve_from_factor, in_axes=(0, 0))(L_right_int, V_mid_T_int)
                VSi_int = jnp.swapaxes(Y_right, -1, -2)

                # Update Sii: Sii - VTS * V_left - VSi * V_mid^T
                Sii_int = Sii_int - jnp.matmul(VTS_int, V_left_int) \
                                - jnp.matmul(VSi_int, V_mid_T_int)
                Sii_int = 0.5 * (Sii_int + jnp.swapaxes(Sii_int, -1, -2))

                # write back caches and S_next
                VTS_inv  = VTS_inv.at[lev, i_left_int].set(VTS_int)
                V_S_inv1 = V_S_inv1.at[lev, i_right_int].set(VSi_int)
                S_levels = S_levels.at[lev + 1, j_int].set(Sii_int)

            # ------------------------------------------------
            # 3) Boundary odd node j = N_next-1
            # ------------------------------------------------
            # This one may not have a right neighbor.
            j_b = N_next - 1
            i_pos_b = int(2 * j_b + 1)

            Sii_b = S_cur[i_pos_b]

            # left neighbor always exists
            L_left_b = S_factors_levels[lev, i_pos_b - 1]
            V_left_b = V_cur[i_pos_b - 1]
            Y_left_b = self.spd_solve_from_factor(L_left_b, V_left_b)
            VTS_b = Y_left_b.T
            VTS_inv = VTS_inv.at[lev, i_pos_b - 1].set(VTS_b)
            Sii_b = Sii_b - VTS_b @ V_left_b

            # optional right neighbor
            if (i_pos_b + 1) < N_cur:
                L_right_b = S_factors_levels[lev, i_pos_b + 1]
                V_mid_b   = V_cur[i_pos_b]
                Y_right_b = self.spd_solve_from_factor(L_right_b, V_mid_b.T)
                VSi_b = Y_right_b.T
                V_S_inv1 = V_S_inv1.at[lev, i_pos_b + 1].set(VSi_b)
                Sii_b = Sii_b - VSi_b @ V_mid_b.T

            Sii_b = 0.5 * (Sii_b + Sii_b.T)
            S_levels = S_levels.at[lev + 1, j_b].set(Sii_b)

            # ------------------------------------------------
            # 4) Build V_next[0..N_next-2] in parallel
            #    V_next[j] = -V_S_inv1[lev, i_pos+1] @ V_cur[i_pos+1]
            # ------------------------------------------------
            V_next = V_levels[lev + 1]
            if N_next > 1:
                j_int = jnp.arange(N_next - 1)
                i_pos_int   = 2 * j_int + 1
                i_upper_int = i_pos_int + 1          # neighbor above each odd
                VSi_int     = V_S_inv1[lev, i_upper_int]      # (N_next-1, n, n)
                V_upper_int = V_cur[i_upper_int]              # (N_next-1, n, n)

                V_next_int = -jnp.matmul(VSi_int, V_upper_int)  # (N_next-1, n, n)
                V_next = V_next.at[j_int].set(V_next_int)

            V_levels = V_levels.at[lev + 1].set(V_next)

        return S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels

    def update_rhs(self, hs, Cs, y, w, rho):
        """
        JAX / JIT-friendly vectorized RHS construction.

        Shapes:
            hs  : (N+1, nv)
            Cs  : (N+1, nc, nv)
            y   : (N+1, nc)
            w   : (N+1, nc)
            rho : scalar
        """

        # (N+1, nc)
        y_minus_w = y - w

        # Compute C_k^T (y_k - w_k) for all k:
        # einsum "kji,kj->ki" performs k-wise matrix-vector product with Cs^T
        Cty = jnp.einsum("kji,kj->ki", Cs, y_minus_w)   # (N+1, nv)

        # g_k = h_k + rho * C_k^T @ (y_k - w_k)
        g = hs + rho * Cty

        return g

    def condense_rhs(self, GF_inv, HTF_inv, g, x0, d_bar):
        """
        Shapes:
            GF_inv  : (N,   nx, nv)     = (G_k F_k^{-1})
            HTF_inv : (N+1, nx, nv)     = (H^T F_k^{-1})
            g       : (N+1, nv)
            x0      : (nx,)
            d_bar   : (N+1, nx)         with d_bar[N]=0
        Returns:
            u       : (N, nx)
        """
        # enforce initial conditioning
        g0_mod = g[0].at[:x0.shape[0]].set(-x0)
        g_mod  = g.at[0].set(g0_mod)
        g_neg  = -g_mod

        # base terms
        term1 = jnp.einsum("kij,kj->ki", GF_inv,      g_neg[:-1])   # (N,nx)
        term2 = jnp.einsum("kij,kj->ki", HTF_inv[1:], g_neg[1:])    # (N,nx)

        u = term1 + term2

        # affine dynamics correction
        u = u + d_bar[:-1]
        return u

    def reduce_to_single_system(self, S_blk, rhs_vec, VTS_inv, V_S_inv1):
        """
        Forward reduction of the block-tridiagonal system to a single coarse system.

        S_blk   : (M, n, n)
        rhs_vec : (M, n)
        VTS_inv : (L_max+1, M,   n, n)
        V_S_inv1: (L_max+1, M,   n, n)

        Returns:
            rhs_levels : (L_max+1, M, n)
            idx_levels : (L_max+1, M)
            M_levels   : (L_max+1,)  (JAX int32, but used *only* as data, not for int())
            last_level : Python int
        """
        M = S_blk.shape[0]
        n = rhs_vec.shape[1]

        # -------- number of levels (pure Python) --------
        L_max = int(math.ceil(math.log2(M)))
        if (M & (M - 1)) == 0:  # power-of-two special case
            L_max += 1

        # Precompute M_levels as Python ints (nodes per level)
        M_levels_py = [max(1, M // (2 ** lev)) for lev in range(L_max + 1)]

        # Allocate level arrays in JAX
        rhs_levels = jnp.zeros((L_max + 1, M, n))
        idx_levels = -jnp.ones((L_max + 1, M), dtype=jnp.int32)

        # Level 0 (original)
        rhs_levels = rhs_levels.at[0, :M].set(rhs_vec)
        idx_levels = idx_levels.at[0, :M].set(jnp.arange(M, dtype=jnp.int32))

        last_level = 0

        # -------- forward reduction over levels (Python loop with static bounds) --------
        for lev in range(L_max):
            N_cur = M_levels_py[lev]
            if N_cur <= 1:
                last_level = lev
                break

            N_next = M_levels_py[lev + 1]

            rhs_cur = rhs_levels[lev]  # (M, n)
            idx_cur = idx_levels[lev]  # (M,)

            rhs_next = rhs_levels[lev + 1]
            idx_next = idx_levels[lev + 1]

            # ---- 1) Interior odd nodes j = 0 .. N_next-2 (vmapped) ----
            if N_next > 1:
                j_int       = jnp.arange(N_next - 1, dtype=jnp.int32)  # (N_next-1,)
                i_pos_int   = 2 * j_int + 1                            # odd positions
                i_left_int  = i_pos_int - 1                            # left neighbor
                i_right_int = i_pos_int + 1                            # right neighbor

                def compute_ri(i_pos_j, i_left_j, i_right_j):
                    ri = rhs_cur[i_pos_j]        # (n,)
                    uL = rhs_cur[i_left_j]       # (n,)
                    uR = rhs_cur[i_right_j]      # (n,)

                    ri = (ri
                        - VTS_inv[lev, i_left_j]  @ uL
                        - V_S_inv1[lev, i_right_j] @ uR)
                    idx_val = idx_cur[i_pos_j]
                    return ri, idx_val

                ris_int, idx_vals_int = jax.vmap(
                    compute_ri,
                    in_axes=(0, 0, 0),
                    out_axes=(0, 0),
                )(i_pos_int, i_left_int, i_right_int)

                rhs_next = rhs_next.at[j_int].set(ris_int)
                idx_next = idx_next.at[j_int].set(idx_vals_int)

            # ---- 2) Last kept odd node j = N_next-1 (scalar) ----
            j_last     = N_next - 1
            i_pos_last = 2 * j_last + 1

            ri = rhs_cur[i_pos_last]
            # left neighbor
            L_pos = i_pos_last - 1
            uL = rhs_cur[L_pos]
            ri = ri - VTS_inv[lev, L_pos] @ uL

            # right neighbor (if exists)
            R_pos = i_pos_last + 1
            if R_pos < N_cur:
                uR = rhs_cur[R_pos]
                ri = ri - V_S_inv1[lev, R_pos] @ uR

            rhs_next = rhs_next.at[j_last].set(ri)
            idx_next = idx_next.at[j_last].set(idx_cur[i_pos_last])

            # write back whole level
            rhs_levels = rhs_levels.at[lev + 1].set(rhs_next)
            idx_levels = idx_levels.at[lev + 1].set(idx_next)

            last_level = lev + 1

        # Convert Python list to JAX array for output (safe; used only as data)
        M_levels = jnp.array(M_levels_py, dtype=jnp.int32)

        return rhs_levels, idx_levels, M_levels, last_level

    def _back_substitute_interior_level(self,
                                        V_l,          # (M_cur, n, n)
                                        rhs_l,        # (M_cur, n)
                                        idx_l,        # (M_cur,)
                                        p_global,     # (M_total, n)
                                        solved,       # (M_total,) bool
                                        S_factors_l): # (M_cur, n, n)
        M_cur = rhs_l.shape[0]
        n = rhs_l.shape[1]

        m_all = jnp.arange(M_cur)               # (M_cur,)

        # valid interior positions: 1 .. M_cur-2
        valid = (m_all > 0) & (m_all < M_cur - 1)

        # global indices for each local position
        g_idx = idx_l                            # (M_cur,)
        g_valid = (g_idx >= 0) & (~solved[g_idx])

        # final mask: interior & unsolved
        mask = valid & g_valid                   # (M_cur,)

        # Neighbor positions (use +1 for right neighbor; clip only for safety)
        L_pos = jnp.clip(m_all - 1, 0, M_cur - 1)
        R_pos = jnp.clip(m_all + 1, 0, M_cur - 1)

        gL = idx_l[L_pos]
        gR = idx_l[R_pos]

        # Neighbor solutions
        pL = p_global[gL]                        # (M_cur, n)
        pR = p_global[gR]                        # (M_cur, n)

        # V blocks
        V_left = V_l[L_pos]                      # (M_cur, n, n)
        V_mid  = V_l[m_all]                      # (M_cur, n, n)

        rhs_eff = rhs_l
        rhs_eff = rhs_eff - jnp.einsum("mij,mj->mi", jnp.swapaxes(V_left, -1, -2), pL)
        rhs_eff = rhs_eff - jnp.einsum("mij,mj->mi", V_mid,                               pR)

        def solve_one(L, b):
            x = self.spd_solve_from_factor(L, b)
            return x.reshape(n,)

        p_candidate = jax.vmap(solve_one, in_axes=(0, 0))(S_factors_l, rhs_eff)  # (M_cur, n)

        # ---- manually mask updates before scatter ----
        old_vals = p_global[g_idx]                           # (M_cur, n)
        new_vals = jnp.where(mask[:, None], p_candidate, old_vals)
        p_global = p_global.at[g_idx].set(new_vals)

        old_flags = solved[g_idx]                            # (M_cur,)
        new_flags = jnp.where(mask, jnp.ones_like(old_flags, bool), old_flags)
        solved   = solved.at[g_idx].set(new_flags)

        return p_global, solved

    def back_substitute_even_odd(self, V_levels, rhs_levels,
                             idx_levels,
                             last_level, solved, p_global, n, S_factors_levels):
        """
        V_levels         : (L_max+1, M_max-1, n, n)
        rhs_levels       : (L_max+1, M_max,   n)
        idx_levels       : (L_max+1, M_max)
        last_level       : Python int
        solved           : (M_total,) bool
        p_global         : (M_total, n)
        S_factors_levels : (L_max+1, M_max, n, n)
        """
        p_global = jnp.array(p_global)
        solved   = jnp.array(solved)

        M_total = rhs_levels.shape[1]

        # walk levels backward: coarsest → finest
        for lev in range(last_level - 1, -1, -1):
            # static formula for number of nodes at this level
            M_cur = max(1, M_total // (2 ** lev))

            V_l   = V_levels[lev, :M_cur]       # (M_cur-1 or M_cur, n, n)
            rhs_l = rhs_levels[lev, :M_cur]     # (M_cur, n)
            idx_l = idx_levels[lev, :M_cur]     # (M_cur,)
            Sf_l  = S_factors_levels[lev, :M_cur]

            # ---------------- Left boundary node (m_pos = 0) ----------------
            g0 = idx_l[0]   # JAX int scalar

            def solve_left_node(carry):
                p_g, s = carry
                rhs_eff = rhs_l[0]

                # Right neighbor m_pos = 1 if it exists
                def with_right(rhs_eff_inner):
                    gR = idx_l[1]
                    rhs_eff_inner = rhs_eff_inner - V_l[0] @ p_g[gR]
                    return rhs_eff_inner

                rhs_eff = lax.cond(
                    M_cur > 1,
                    with_right,
                    lambda r: r,
                    rhs_eff,
                )

                p0 = self.spd_solve_from_factor(Sf_l[0], rhs_eff)
                p_g = p_g.at[g0].set(p0)
                s   = s.at[g0].set(True)
                return p_g, s

            do_left = (g0 >= 0) & (~solved[g0])
            p_global, solved = lax.cond(
                do_left,
                solve_left_node,
                lambda carry: carry,
                operand=(p_global, solved),
            )

            # ---------------- Interior nodes (vmapped) ----------------
            if M_cur > 2:
                p_global, solved = self._back_substitute_interior_level(
                    V_l, rhs_l, idx_l, p_global, solved, Sf_l
                )

            # ---------------- Right boundary node (m_pos = M_cur-1) ----------------
            if M_cur > 1:
                m_last = M_cur - 1
                g_last = idx_l[m_last]

                def solve_right_node(carry):
                    p_g, s = carry
                    rhs_eff = rhs_l[m_last]

                    # Left neighbor m_pos = M_cur-2
                    L_pos = m_last - 1
                    gL = idx_l[L_pos]
                    rhs_eff = rhs_eff - V_l[L_pos].T @ p_g[gL]

                    p_last = self.spd_solve_from_factor(Sf_l[m_last], rhs_eff)
                    p_g = p_g.at[g_last].set(p_last)
                    s   = s.at[g_last].set(True)
                    return p_g, s

                do_right = (g_last >= 0) & (~solved[g_last])
                p_global, solved = lax.cond(
                    do_right,
                    solve_right_node,
                    lambda carry: carry,
                    operand=(p_global, solved),
                )

        return p_global

    def solve_p_even_odd(self, S_blk, rhs_vec,
                         S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels):
        """
        JAX / JIT-friendly wrapper around odd-even cyclic reduction solve.

        Inputs:
            S_blk           : (M, n, n)
            rhs_vec         : (M, n)
            S_levels        : (L_max+1, M, n, n)
            V_levels        : (L_max+1, M-1, n, n)
            VTS_inv         : (L_max+1, M,   n, n)
            V_S_inv1        : (L_max+1, M,   n, n)
            S_factors_levels: (L_max+1, M,   n, n)

        Returns:
            p_global        : (M, n)
        """
        M = S_blk.shape[0]
        # prefer a static nx if you have it (for jit)
        n = S_blk.shape[1]  # instead of S_blk.shape[1]

        # Forward reduction of RHS to coarsest system
        rhs_levels, idx_levels, M_levels, last_level = self.reduce_to_single_system(
            S_blk, rhs_vec, VTS_inv, V_S_inv1
        )

        # ---------- Base solve on coarsest level ----------
        S_last  = S_levels[last_level]      # (M_last, n, n)
        rhs_last = rhs_levels[last_level]   # (M_last, n)
        idx_last = idx_levels[last_level]   # (M_last,)

        # Global index of the remaining node at the coarsest level
        g0 = idx_last[0]

        # rhs_last[0] is (n,), we need (n,1) for spd_solve
        rhs0 = rhs_last[0].reshape(n, 1)
        p0 = self.spd_solve(S_last[0], rhs0).reshape(n)

        p_global = jnp.zeros((M, n))
        solved   = jnp.zeros((M,), dtype=bool)

        p_global = p_global.at[g0].set(p0)
        solved   = solved.at[g0].set(True)

        # ---------- Backward substitution: recover eliminated nodes ----------
        p_global = self.back_substitute_even_odd(
            V_levels, rhs_levels,
            idx_levels, 
            last_level, solved, p_global, n, S_factors_levels
        )

        return p_global

    def back_substitute(self, z, F, g, Gs, H, GF_inv, HTF_inv, F_chol, N, nx, nu, x0):
        """
        JAX-friendly back-substitution.

        z      : (N,   nx)   solution in Schur variables
        F      : (N+1, nv, nv)
        g      : (N+1, nv)
        Gs     : (N,   nx, nv)
        H      : (nv, nx)          # (unused here, same as original)
        GF_inv : (N,   nx, nv)
        HTF_inv: (N+1, nx, nv)
        F_chol : (N+1, nv, nv)
        N, nx, nu : ints
        x0     : (nx,)
        """
        nv = nx + nu

        # ---------- k = 0 (special: x0 fixed) ----------
        g0 = g[0]
        g0 = g0.at[:nx].set(-x0)        # enforce x_0 = x0

        G0T = Gs[0].T                   # (nv, nx)
        G0T = G0T.at[:nx, :nx].set(0.0) # zero out x_0 part in G

        F0 = F[0]
        F0 = F0.at[:nx, :].set(0.0)
        F0 = F0.at[:, :nx].set(0.0)
        F0 = F0.at[:nx, :nx].set(jnp.eye(nx))

        # b0 = -g0 - G0T @ z0
        b0 = -(g0 + G0T @ z[0])                # (nv,)
        y0 = self.spd_solve(F0, b0.reshape(nv, 1)).reshape(nv,)  # (nv,)

        # ---------- 1 <= k <= N-1 (vectorized) ----------
        if N > 1:
            g_mid       = g[1:N]            # (N-1, nv)
            Fchol_mid   = F_chol[1:N]       # (N-1, nv, nv)
            GF_mid      = GF_inv[1:N]       # (N-1, nx, nv)
            HTF_mid     = HTF_inv[1:N]      # (N-1, nx, nv)
            z_k         = z[1:N]            # (N-1, nx)
            z_prev      = z[0:N-1]          # (N-1, nx)

            def solve_mid(Fc, gk, GFk, HTFk, zk, zkm1):
                bk = -gk                                  # (nv,)
                y_free = self.spd_solve_from_factor(
                    Fc, bk)# (nv,)

                # GFk: (nx, nv) or (nx, nv)? In your code GF_inv[k] has shape (nx, nv)
                # so GFk.T: (nv, nx)
                corr = (GFk.T @ zk) + (HTFk.T @ zkm1)     # (nv,)
                return y_free - corr                      # (nv,)

            ys_mid = jax.vmap(
                solve_mid,
                in_axes=(0, 0, 0, 0, 0, 0)
            )(Fchol_mid, g_mid, GF_mid, HTF_mid, z_k, z_prev)  # (N-1, nv)
        else:
            ys_mid = jnp.zeros((0, nv))

        # ---------- k = N (terminal) ----------
        gN = g[N]
        bN = -gN
        yN_free = self.spd_solve_from_factor(
            F_chol[N], bN
        )

        yN = yN_free - HTF_inv[N].T @ z[N - 1]  # (nv,)

        # ---------- Stack all ----------
        ys = jnp.vstack([y0[None, :], ys_mid, yN[None, :]])  # (N+1, nv)
        return ys

    def solve_system(self, N, nx, nu,
                     hs, As, Cs,
                     y, w, rho, x0, d_bar,
                     F, GF_inv, HTF_inv, F_chol, H,
                     S,
                     S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels):
        g = self.update_rhs(hs, Cs, y, w, rho)

        # First Layer of Condensing
        u = self.condense_rhs(GF_inv, HTF_inv, g, x0, d_bar)
        solve = self.solve_p_even_odd(S, u, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels)


        # Back-substitute to recover (x,u)
        sol = self.back_substitute(solve, F, g, As, H, GF_inv, HTF_inv, F_chol, N, nx, nu, x0)
        return sol

    # ------- ADMM Helpers -------
    def calculate_scaling(self, z_plus, y_plus, Hs, hs, Cs):
        """
        JAX version, JIT-friendly.

        Shapes:
            z_plus : (N+1, nv)
            y_plus : (N+1, nc)
            Hs     : (N+1, nv, nv)
            hs     : (N+1, nv)
            Cs     : (N+1, nc, nv)
        """
        # Hy_k = Hs_k @ z_k
        Hy     = jnp.einsum("kij,kj->ki", Hs, z_plus)      # (N+1, nv)
        # Gy_k = Cs_k @ z_k
        Gy     = jnp.einsum("kij,kj->ki", Cs, z_plus)      # (N+1, nc)
        # GTlam_k = Cs_k^T @ y_k  => sum_i Cs[k, i, j] * y[k, i]
        GTlam  = jnp.einsum("kji,kj->ki", Cs, y_plus)      # (N+1, nv)
        g_vecs = hs                                         # (N+1, nv)

        # Flatten everything
        Hy_vec    = Hy.reshape(-1)
        Gy_vec    = Gy.reshape(-1)
        GTlam_vec = GTlam.reshape(-1)
        g_vec     = g_vecs.reshape(-1)
        z_vec     = z_plus.reshape(-1)

        # Infinity norms
        Hy_inf    = jnp.linalg.norm(Hy_vec,    ord=jnp.inf)
        Gy_inf    = jnp.linalg.norm(Gy_vec,    ord=jnp.inf)
        GTlam_inf = jnp.linalg.norm(GTlam_vec, ord=jnp.inf)
        g_inf     = jnp.linalg.norm(g_vec,     ord=jnp.inf)
        z_inf     = jnp.linalg.norm(z_vec,     ord=jnp.inf)

        return Hy_inf, Gy_inf, GTlam_inf, g_inf, z_inf

    def calculate_rho_scaling(self,
                              rp_inf, rd_inf,
                              Hy_inf, GTlam_inf, g_inf, Gy_inf, z_inf,
                              eps=1e-8):
        """
        JAX-friendly version.

        Inputs are assumed to be scalar jnp arrays or Python floats.
        """
        eps = jnp.asarray(eps)

        # If either residual is exactly zero -> factor = 1.0
        zero_res = jnp.logical_or(rp_inf == 0.0, rd_inf == 0.0)

        # Primary and dual residual magnitudes, clamped below by eps
        pri_res  = jnp.maximum(rp_inf, eps)
        dual_res = jnp.maximum(rd_inf, eps)

        # Normalization scales (max over given norms, then eps)
        pri_res_norm  = jnp.maximum(jnp.maximum(Gy_inf, z_inf), eps)
        dual_res_norm = jnp.maximum(jnp.maximum(g_inf, GTlam_inf), Hy_inf)
        dual_res_norm = jnp.maximum(dual_res_norm, eps)

        pri_scaled  = pri_res  / pri_res_norm
        dual_scaled = dual_res / dual_res_norm

        # If dual_scaled < eps -> factor = 1.0
        too_small_dual = dual_scaled < eps

        # Base factor = sqrt(pri_scaled / dual_scaled)
        # (numerically safe because we clamped by eps above)
        raw_factor = jnp.sqrt(pri_scaled / dual_scaled)

        # Apply the two "return 1.0" conditions
        use_one = jnp.logical_or(zero_res, too_small_dual)
        factor  = jnp.where(use_one, 1.0, raw_factor)

        return factor

    def update_rho(self,
                   rp_inf, rd_inf, rho,
                   Hy_inf, GTlam_inf, g_inf, Gy_inf, z_inf):
        # factor is assumed to be JAX-friendly already
        factor = self.calculate_rho_scaling(
            rp_inf, rd_inf,
            Hy_inf, GTlam_inf, g_inf, Gy_inf, z_inf
        )

        # Clamp rho within [0.01, 100] in JAX
        rho_proposed = rho * factor
        new_rho = jnp.clip(rho_proposed, 0.01, 100.0)

        new_factor = new_rho / rho

        # Decide if change is "significant": outside [0.9, 1.1]
        updated = jnp.logical_or(new_factor < 0.9, new_factor > 1.1)

        # If not significant, keep old rho; otherwise use new_rho
        rho_out = jnp.where(updated, new_rho, rho)

        return rho_out, updated

    def feasible_projection(self, z_plus, y, Cs, f):
        """
        JAX version.

        Shapes:
            z_plus : (N+1, nv)
            y      : (N+1, nc)
            Cs     : (N+1, nc, nv)
            f      : (N+1, nc) or (N+1, nc, 1)
        """
        # h_k = Cs_k @ z_k  for all k
        h = jnp.einsum("kij,kj->ki", Cs, z_plus)   # (N+1, nc)

        # Flatten f if it has a trailing singleton dim
        f_flat = jnp.squeeze(f, axis=-1) if f.ndim == 3 else f  # (N+1, nc)

        # w_k = min( h_k + y_k, f_k )
        w_plus = jnp.minimum(h + y, f_flat)        # (N+1, nc)

        return w_plus

    def update_dual(self, z_plus, y, w_plus, Cs):
        """
        JAX version.

        Shapes:
            z_plus : (N+1, nv)
            y      : (N+1, nc)
            w_plus : (N+1, nc)
            Cs     : (N+1, nc, nv)
        """
        # h_k = Cs_k @ z_k for all k
        h = jnp.einsum("kij,kj->ki", Cs, z_plus)   # (N+1, nc)

        # y_{k}^{new} = y_k + (h_k - w_plus_k)
        y_plus = y + (h - w_plus)                  # (N+1, nc)

        return y_plus

    def calculate_residuals(self, z_plus, Cs, rho, w_plus, w_prev):
        """
        JAX version.

        Shapes:
            z_plus : (N+1, nv)
            Cs     : (N+1, nc, nv)
            w_plus : (N+1, nc)
            w_prev : (N+1, nc)
            rho    : scalar
        """
        # Primal residual: r_prim[k] = Cs[k] @ z_plus[k] - w_plus[k]
        r_prim = jnp.einsum("kij,kj->ki", Cs, z_plus) - w_plus   # (N+1, nc)

        # Dual residual: r_dual[k] = rho * Cs[k]^T @ (w_plus[k] - w_prev[k])
        w_diff = w_plus - w_prev                                 # (N+1, nc)
        # Cs^T @ w_diff: use "kji" to transpose last two axes of Cs in einsum
        r_dual = rho * jnp.einsum("kji,kj->ki", Cs, w_diff)      # (N+1, nv)

        # Infinity norms over all entries
        rp_inf = jnp.linalg.norm(r_prim.reshape(-1), ord=jnp.inf)
        rd_inf = jnp.linalg.norm(r_dual.reshape(-1), ord=jnp.inf)

        return rp_inf, rd_inf

    def compute_osqp_tolerances(self,
                            z_plus,   # (N+1, nv)
                            y_plus,   # (N+1, nc)
                            w_plus,   # (N+1, nc)
                            Hs,       # (N+1, nv, nv)
                            hs,       # (N+1, nv)
                            Cs):      # (N+1, nc, nv)
        """
        Compute OSQP-style primal and dual tolerances:

            eps_pri = eps_abs + eps_rel * max(||C z||_inf, ||w||_inf)
            eps_dual = eps_abs + eps_rel * max(||H z||_inf, ||C^T y||_inf, ||h||_inf)

        where Hs, hs, Cs, z_plus, y_plus come from the condensed QP.
        """
        # Reuse your existing scaling norms
        Hy_inf, Gy_inf, GTlam_inf, g_inf, z_inf = self.calculate_scaling(
            z_plus, y_plus, Hs, hs, Cs
        )

        # Infinity norm of w_plus
        w_inf = jnp.linalg.norm(w_plus.reshape(-1), ord=jnp.inf)

        # Primal tolerance: use Cz vs w (OSQP uses A x vs z)
        eps_pri = self.eps_abs + self.eps_rel * jnp.maximum(Gy_inf, w_inf)

        # Dual tolerance: gradient-related scales (OSQP uses P x, A^T y, q)
        eps_dual = self.eps_abs + self.eps_rel * jnp.maximum(
            jnp.maximum(Hy_inf, GTlam_inf),
            g_inf,
        )

        return eps_pri, eps_dual

    # ------- Solve Loop JIT Helpers -------
    def _admm_cond(self, state):
        (i, y, w, z, rho,
         rp, rd,
         eps_pri, eps_dual,
         F, GF_inv, HTF_inv, F_chol, d_bar,
         S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels) = state

        not_converged = jnp.logical_or(rp > eps_pri, rd > eps_dual)
        under_max_iter = i < self.max_iterations
        return jnp.logical_and(under_max_iter, not_converged)

    def _admm_body(self, state, static):
        (i, y, w, z, rho, rp, rd,
         eps_pri, eps_dual,
         F, GF_inv, HTF_inv, F_chol, d_bar,
         S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels) = state

        (Hs, hs, As, Cs, f_eff,
         N, nx, nu, nc,
         H, x0) = static

        # ----------------------------------------------------
        # 1) Optional rho update (and refactorization)
        # ----------------------------------------------------
        do_rho_update = jnp.logical_and(
            i > 0,
            jnp.equal(jnp.mod(i, self.rho_update_frequency), 0)
        )

        def rho_update_branch(args):
            (y, rho, F, GF_inv, HTF_inv, F_chol, d_bar,
             S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels) = args

            # Scaling norms
            Hy_inf, Gy_inf, GTlam_inf, g_inf, z_inf = self.calculate_scaling(
                z, y, Hs, hs, Cs
            )

            # Propose new rho
            rho_new, updated = self.update_rho(
                rp, rd, rho,
                Hy_inf, GTlam_inf, g_inf, Gy_inf, z_inf
            )

            # Safeguard rho > 0
            rho_safe = jnp.where(rho_new <= 0.0, rho, rho_new)

            # Scale duals if rho actually changed
            scale = rho / rho_safe
            do_scale = jnp.logical_and(updated, rho_safe > 0.0)
            y_scaled = jnp.where(do_scale, y * scale, y)

            def recompute_mats(_):
                F_new = self.assemble_F(Hs, Cs, rho_safe, N)
                GF_inv_new, HTF_inv_new, F_chol_new = self.assemble_first_schur(
                    F_new, As, H, N, nx, nu
                )
                S_new, V_new = self.condense_cache(
                    F_new, GF_inv_new, HTF_inv_new, As, H, N, nx
                )
                (S_levels_new, V_levels_new, VTS_inv_new,
                 V_S_inv1_new, S_factors_levels_new) = self.reduce_to_single_system_caches(
                    S_new, V_new
                )
                return (y_scaled, rho_safe, F_new, GF_inv_new, HTF_inv_new, F_chol_new, d_bar,
                        S_new, S_levels_new, V_levels_new, VTS_inv_new, V_S_inv1_new,
                        S_factors_levels_new)

            def keep_mats(_):
                # rho_safe already encodes "no change" when rho_new <= 0
                return (y_scaled, rho_safe, F, GF_inv, HTF_inv, F_chol, d_bar,
                        S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels)

            return lax.cond(
                updated,
                recompute_mats,
                keep_mats,
                operand=None
            )

        def rho_no_update_branch(args):
            (y, rho, F, GF_inv, HTF_inv, F_chol, d_bar,
             S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels) = args
            return (y, rho, F, GF_inv, HTF_inv, F_chol, d_bar,
                    S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels)

        (y, rho, F, GF_inv, HTF_inv, F_chol, d_bar,
         S, S_levels, V_levels, VTS_inv, V_S_inv1,
         S_factors_levels) = lax.cond(
            do_rho_update,
            rho_update_branch,
            rho_no_update_branch,
            operand=(y, rho, F, GF_inv, HTF_inv, F_chol, d_bar,
                     S, S_levels, V_levels, VTS_inv, V_S_inv1,
                     S_factors_levels),
        )

        # ----------------------------------------------------
        # 2) Primal update: solve condensed system for z
        # ----------------------------------------------------
        z_plus = self.solve_system(
            N, nx, nu,
            hs, As, Cs,
            y, w, rho, x0, d_bar,
            F, GF_inv, HTF_inv, F_chol, H,
            S,
            S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels,
        )

        # ----------------------------------------------------
        # 3) Projection and dual update
        # ----------------------------------------------------
        w_prev = w

        no_constr = (nc == 0)

        def proj_dual_no_constr(_):
            # Keep shapes consistent with the constrained branch:
            # w lives in constraint space, so keep it as-is (or zeros_like).
            w_plus = w
            y_plus = jnp.zeros_like(y)
            return w_plus, y_plus

        def proj_dual_with_constr(_):
            w_plus = self.feasible_projection(z_plus, y, Cs, f_eff)
            y_plus = self.update_dual(z_plus, y, w_plus, Cs)
            return w_plus, y_plus

        w_plus, y_plus = lax.cond(no_constr, proj_dual_no_constr, proj_dual_with_constr, operand=None)

        # ----------------------------------------------------
        # 4) Residuals
        # ----------------------------------------------------
        rp_new, rd_new = self.calculate_residuals(
            z_plus, Cs, rho, w_plus, w_prev
        )

        eps_pri_new, eps_dual_new = self.compute_osqp_tolerances(
            z_plus,        # current primal variable
            y_plus,        # current dual variable
            w_plus,        # current splitting / projection
            Hs, hs, Cs,    # condensed QP data
        )

        # ----------------------------------------------------
        # 5) Next state
        # ----------------------------------------------------
        next_state = (
            i + 1,
            y_plus,
            w_plus,
            z_plus,
            rho,
            rp_new,
            rd_new,
            eps_pri_new,
            eps_dual_new,
            F, GF_inv, HTF_inv, F_chol, d_bar,
            S, S_levels, V_levels, VTS_inv, V_S_inv1, S_factors_levels,
        )

        # jax.debug.print(
        #     "ADMM iter {i}: prim={p:.3e} (tol={tp:.3e}), dual={d:.3e} (tol={td:.3e}), rho={r:.3e}",
        #     i=i,
        #     p=rp_new,
        #     tp=eps_pri_new,
        #     d=rd_new,
        #     td=eps_dual_new,
        #     r=rho
        # )
        return next_state

        # ---------------------------
    # Preconditioning / Scaling
    # ---------------------------

    def _safe_inv_sqrt(self, x, eps=1e-12):
        x = jnp.maximum(x, eps)
        return jax.lax.rsqrt(x)

    def compute_global_variable_scaling(self, Hs, Cs, eps=1e-12, clip=(1e-3, 1e3)):
        """
        Compute a SINGLE diagonal variable scaling D (nv,) shared across all stages.
        Uses max(abs(.)) across columns of H and C.

        Returns:
          d    : (nv,) so that v = diag(d) * v_hat
          dinv : (nv,) = 1/d
        """
        # Hs: (N+1, nv, nv) -> per-column max abs across stages and rows
        H_col = jnp.max(jnp.abs(Hs), axis=(0, 1))  # (nv,)

        # Cs: (N+1, nc, nv) -> per-column max abs across stages and rows
        if Cs.size == 0:
            C_col = jnp.zeros_like(H_col)
        else:
            C_col = jnp.max(jnp.abs(Cs), axis=(0, 1))  # (nv,)

        col_scale = jnp.maximum(H_col, C_col)  # (nv,)
        dinv = self._safe_inv_sqrt(col_scale, eps=eps)
        dinv = jnp.clip(dinv, clip[0], clip[1])
        d = 1.0 / dinv
        return d, dinv

    def compute_constraint_row_scaling(self, Cs_var, eps=1e-12, clip=(1e-3, 1e3)):
        """
        Per-stage row scaling E_k for inequalities.
        Cs_var: (N+1, nc, nv) already includes variable scaling: C*D.

        Returns:
          e_rows: (N+1, nc) multipliers; E_k = diag(e_rows[k])
        """
        if Cs_var.size == 0:
            return jnp.zeros((Cs_var.shape[0], 0), dtype=Cs_var.dtype)

        row_inf = jnp.max(jnp.abs(Cs_var), axis=-1)  # (N+1, nc)

        # IMPORTANT: avoid scaling "inactive" all-zero rows (common in your terminal block padding)
        active = row_inf > 0
        row_inf = jnp.where(active, row_inf, 1.0)

        einv = self._safe_inv_sqrt(row_inf, eps=eps)
        einv = jnp.clip(einv, clip[0], clip[1])
        e_rows = 1.0 / einv
        return e_rows

    def apply_scaling(self, Hs, hs, As, Cs, f_eff, d, e_rows):
        """
        Apply global variable scaling d (nv,) and per-stage constraint row scaling e_rows (N+1,nc).

        v = D v_hat  => H_hat = D^T H D, h_hat = D^T h, A_hat = A D, C_hat = E C D, f_hat = E f
        """
        # H_hat[i,j] = d[i]*H[i,j]*d[j]
        Hs_hat = (Hs * d[None, None, :]) * d[None, :, None]
        hs_hat = hs * d[None, :]

        # Dynamics coupling uses A v, so A_hat = A D
        As_hat = As * d[None, None, :]

        if Cs.size == 0:
            return Hs_hat, hs_hat, As_hat, Cs, f_eff

        # Variable scaling on constraints: C_var = C D
        Cs_var = Cs * d[None, None, :]

        # Row scaling: C_hat = E C_var, f_hat = E f
        Cs_hat = Cs_var * e_rows[:, :, None]

        if f_eff.ndim == 3:
            f_hat = f_eff * e_rows[:, :, None]
        else:
            f_hat = f_eff * e_rows

        return Hs_hat, hs_hat, As_hat, Cs_hat, f_hat

    def make_scaled_dynamics_selector(self, nx, nu, d):
        """
        Your unscaled selector was H = [-I; 0] acting on v=[x;u].

        With x = d_x * x_hat, enforcing x0 in scaled coordinates requires:
          (-I) x = -x0  => (-diag(d_x)) x_hat = -x0

        Returns H_hat: (nv, nx)
        """
        d_x = d[:nx]
        Hx = -jnp.diag(d_x)
        Hu = jnp.zeros((nu, nx), dtype=d.dtype)
        return jnp.concatenate([Hx, Hu], axis=0)

    def x0_to_scaled(self, x0, d, nx):
        """x = d_x * x_hat => x_hat = x / d_x."""
        return x0 / d[:nx]

    def z_orig_to_scaled(self, z_orig, d):
        """v = D v_hat => v_hat = v / d."""
        return z_orig / d[None, :]

    def z_scaled_to_orig(self, z_scaled, d):
        """v = D v_hat => v = v_hat * d."""
        return z_scaled * d[None, :]

    def yw_orig_to_scaled(self, y_orig, w_orig, e_rows):
        """
        If constraints are scaled as: C_hat = E C D, f_hat = E f
        then dual scaling consistent with residuals is:
          y_hat = y_orig / E
        and the splitting/projection variable (in constraint space) should satisfy:
          w_hat = E w_orig
        """
        if y_orig.size == 0:
            return y_orig, w_orig
        return y_orig / e_rows, w_orig * e_rows

    def yw_scaled_to_orig(self, y_scaled, w_scaled, e_rows):
        """Inverse mapping: y_orig = E y_hat, w_orig = w_hat / E."""
        if y_scaled.size == 0:
            return y_scaled, w_scaled
        return y_scaled * e_rows, w_scaled / e_rows
    
    def symmetrize(self, H):
        return 0.5 * (H + jnp.swapaxes(H, -1, -2))

    def regularize_hessian(self, H, rel=1e-6, abs_=1e-8, min_diag=1e-12):
        H = self.symmetrize(H)
        d = jnp.abs(jnp.diagonal(H, axis1=-2, axis2=-1))
        d_mean = jnp.maximum(jnp.mean(d, axis=-1, keepdims=True), min_diag)
        delta = abs_ + rel * d_mean
        n = H.shape[-1]
        I = jnp.eye(n, dtype=H.dtype)
        return H + delta[..., None] * I

    def regularize_hessian_adaptive(
        self,
        H,
        mu: float = 1e-4,          # desired curvature floor (trust-region strength)
        jitter_min: float = 1e-9,
        jitter_max: float = 1e2,
    ):
        """
        Adaptive, spectrum-aware Hessian regularization.

        Guarantees (approximately):
            H_reg ≽ mu * I

        JAX-safe:
        - no eigenvalues
        - no Python branching
        - no exceptions
        """
        # Ensure symmetry
        H = 0.5 * (H + jnp.swapaxes(H, -1, -2))

        # Gershgorin lower bound on lambda_min
        abs_H = jnp.abs(H)
        row_sum = jnp.sum(abs_H, axis=-1) - jnp.abs(jnp.diagonal(H, axis1=-2, axis2=-1))
        gersh_lb = jnp.diagonal(H, axis1=-2, axis2=-1) - row_sum

        # Worst-case lower bound
        lambda_lb = jnp.min(gersh_lb, axis=-1)   # (...,)

        # Required shift so that lambda_min >= mu
        delta = jnp.maximum(mu - lambda_lb, 0.0)

        # Clamp for numerical sanity
        delta = jnp.clip(delta, jitter_min, jitter_max)

        n = H.shape[-1]
        I = jnp.eye(n, dtype=H.dtype)

        return H + delta[..., None] * I

    def symmetrize(self, H):
       return 0.5 * (H + jnp.swapaxes(H, -1, -2))

    def regularize_hessian_eig_floor(
        self,
        H: jax.Array,
        eta: float = 1e-4,       # desired min eigenvalue after regularization (trust-region strength)
        delta_min: float = 0.0,  # always add at least this much (optional)
        delta_max: float = 1e2,  # safety clamp
    ):
        """
        Convexify H by shifting the diagonal so that lambda_min(H_reg) >= eta.

        JIT-safe:
        - uses jnp.linalg.eigvalsh (works under jit)
        - no exceptions
        - deterministic

        Note: For float32, eigvalsh is stable enough for moderate nv (typical condensed blocks).
        """
        Hs = self.symmetrize(H)

        # Smallest eigenvalue (symmetric eigensolver)
        w = jnp.linalg.eigvalsh(Hs)          # (n,)
        lam_min = w[0]

        # Required diagonal shift
        delta = jnp.maximum(delta_min, eta - lam_min)
        delta = jnp.clip(delta, 0.0, delta_max)

        n = H.shape[-1]
        return Hs + delta * jnp.eye(n, dtype=H.dtype), delta, lam_min

    @partial(jax.jit, static_argnums=(0,))
    def _solve_core(self,
                    problem: ADMMProblem,
                    x0,
                    y0_init_orig, w0_init_orig, z0_init_orig, rho0_init):
        # Original problem
        N_orig = problem.N
        Q, q, R, r = problem.Q, problem.q, problem.R, problem.r
        C, Dm, f = problem.C, problem.D, problem.f
        A, B, c = problem.A, problem.B, problem.c
        nx, nu_orig, nc_orig = problem.nx, problem.nu, problem.nc
        m = self.condense_block_size

        # Build partially condensed problem (UNSCALED)
        (Hs, hs, As, d_bar, Cs, f_eff,
         N_eff, nv_eff, nc_eff) = self.condense_problem_partial(
            Q, q, R, r, A, B, c, C, Dm, f,
            N_orig, nx, nu_orig, nc_orig, m
        )
        Hs = jax.vmap(lambda H: self.regularize_hessian(H, rel=1e-4, abs_=1e-6))(Hs)
        nu_eff = nv_eff - nx

        # -------------------------------
        # Precondition: global variable scaling D + per-stage constraint row scaling E_k
        # -------------------------------
        d, dinv = self.compute_global_variable_scaling(Hs, Cs)
        if nc_eff == 0:
            e_rows = jnp.zeros((N_eff + 1, 0), dtype=Hs.dtype)
        else:
            Cs_var = Cs * d[None, None, :]
            e_rows = self.compute_constraint_row_scaling(Cs_var)

        Hs, hs, As, Cs, f_eff = self.apply_scaling(Hs, hs, As, Cs, f_eff, d, e_rows)
        # Hs = jax.vmap(lambda H: self.regularize_hessian(H, rel=1e-4, abs_=1e-6))(Hs)
        # Hs = jax.vmap(lambda H: self.regularize_hessian_eig_floor(H, eta=1e-2)[0])(Hs)

        # Work on condensed problem
        N = N_eff
        nu = nu_eff
        nc = nc_eff

        # Scale x0 into scaled coordinates
        x0_scaled = self.x0_to_scaled(x0, d, nx)

        # Convert warm-start initial values (ORIG coords) -> (SCALED coords)
        z0 = self.z_orig_to_scaled(z0_init_orig, d)
        if nc > 0:
            y0, w0 = self.yw_orig_to_scaled(y0_init_orig, w0_init_orig, e_rows)
        else:
            y0, w0 = y0_init_orig, w0_init_orig

        rho0 = rho0_init

        rp0 = jnp.inf
        rd0 = jnp.inf

        # Scaled dynamics selector for Schur construction
        H = self.make_scaled_dynamics_selector(nx, nu, d)

        # Initial factorization with given rho0
        F0 = self.assemble_F(Hs, Cs, rho0, N)
        GF_inv0, HTF_inv0, F_chol0 = self.assemble_first_schur(
            F0, As, H, N, nx, nu
        )
        S0, V0 = self.condense_cache(F0, GF_inv0, HTF_inv0, As, H, N, nx)
        (S_levels0, V_levels0, VTS_inv0,
         V_S_inv10, S_factors_levels0) = self.reduce_to_single_system_caches(S0, V0)

        eps_pri0  = jnp.array(0.0)
        eps_dual0 = jnp.array(0.0)

        state0 = (
            jnp.array(0),
            y0,
            w0,
            z0,
            jnp.array(rho0),
            jnp.array(rp0),
            jnp.array(rd0),
            jnp.array(eps_pri0),
            jnp.array(eps_dual0),
            F0, GF_inv0, HTF_inv0, F_chol0, d_bar,
            S0, S_levels0, V_levels0, VTS_inv0, V_S_inv10, S_factors_levels0,
        )

        # Things that don't change go into "static"
        static = (
            Hs, hs, As, Cs, f_eff,
            N, nx, nu, nc,
            H, x0_scaled,
        )

        def cond_fun(state):
            return self._admm_cond(state)

        def body_fun(state):
            return self._admm_body(state, static)

        final_state = lax.while_loop(cond_fun, body_fun, state0)

        (i_final, y_final, w_final, z_final, rho_final,
         rp_final, rd_final,
         eps_pri_final, eps_dual_final,
         F_final, GF_inv_final, HTF_inv_final, F_chol_final, d_bar_final,
         S_final, S_levels_final, V_levels_final, VTS_inv_final, V_S_inv1_final,
         S_factors_levels_final) = final_state

        # -------------------------------
        # Unscale solution back to ORIGINAL condensed coordinates
        # -------------------------------
        z_final_orig = self.z_scaled_to_orig(z_final, d)

        if nc > 0:
            y_final_orig, w_final_orig = self.yw_scaled_to_orig(y_final, w_final, e_rows)
        else:
            y_final_orig, w_final_orig = y_final, w_final

        # ----- Reconstruct trajectories on original time grid -----
        x_full, u_full = self.unpartially_condense(
            m, z_final_orig, nu_orig, nx, N_eff, x0, problem.A, problem.B, problem.c
        )

        # jax.debug.print(
        #     "ADMM iterations {i}: prim={p:.3e}, dual={d:.3e}, rho={r:.3e}",
        #     i=i_final,
        #     p=rp_final,
        #     d=rd_final,
        #     r=rho_final
        # )

        # Return full solution + ORIGINAL-space warm-start buffers
        return x_full, u_full, z_final_orig, y_final_orig, w_final_orig, rho_final


    def solve(self, problem: ADMMProblem, x0: Array, warm_start: bool = True):
        """
        Public entry point with optional warm start.

        IMPORTANT: warm-start buffers are stored in ORIGINAL condensed coordinates.
        Each call will compute fresh scalings internally and convert as needed.
        """
        N_orig = problem.N
        nx, nu_orig, nc_orig = problem.nx, problem.nu, problem.nc
        m = self.condense_block_size

        # Condensed dimensions (match condense_problem_partial)
        N_blocks = N_orig // m
        N_eff = N_blocks
        nv_eff = nx + m * nu_orig
        nc_eff = m * nc_orig

        can_ws = False
        if warm_start and (self._z_ws is not None):
            same_z = (self._z_ws.shape == (N_eff + 1, nv_eff))
            same_w = (self._w_ws is not None and self._w_ws.shape == (N_eff + 1, nc_eff))
            same_y = (self._y_ws is not None and self._y_ws.shape == (N_eff + 1, nc_eff))
            have_rho = (self._rho_ws is not None)
            can_ws = same_z and same_w and same_y and have_rho

        if can_ws:
            # ORIGINAL-space warm start buffers
            z0_orig = self._z_ws
            w0_orig = self._w_ws
            y0_orig = self._y_ws
            rho0 = jnp.asarray(self._rho_ws)
        else:
            # ORIGINAL-space initialization
            z0_orig = jnp.zeros((N_eff + 1, nv_eff))
            w0_orig = jnp.zeros((N_eff + 1, nc_eff))
            y0_orig = jnp.zeros((N_eff + 1, nc_eff))
            rho0 = jnp.asarray(0.1)

        (x_full, u_full,
         z_final_orig, y_final_orig, w_final_orig, rho_final) = self._solve_core(
            problem, x0,
            y0_orig, w0_orig, z0_orig, rho0
        )

        # Store warm-start state for next call (ORIGINAL condensed coordinates)
        self._z_ws = z_final_orig
        self._w_ws = w_final_orig
        self._y_ws = y_final_orig
        self._rho_ws = rho_final

        return x_full, u_full
