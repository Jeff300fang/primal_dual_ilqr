"""
End-to-end reference script:

1) Build dummy TVLQR-style scan elements (A, c, C, p, P) in JAX packed format: (T, 3n+2, n)
2) Build reduced elements (A, C, P) in NumPy packed format: (T, 3n, n)
3) Run NumPy associative scan on (A,C,P) while caching per-level factors:
      Ar = A_r (I + C_l P_r)^-1
      Al = A_l^T (I + P_r C_l)^-1
      ArC = Ar C_l
      AlP = Al P_r
      plus C_new, P_new (optional, but often useful)
4) Use cached factors to propagate c and p (no inverses), reproducing JAX's scan output
5) Compare NumPy cached results to JAX lax.associative_scan(vmap(fn)) results

This matches the TVLQR paper-style operator order with reverse=True.
"""

import numpy as np
import jax
import jax.numpy as jnp


# -----------------------------
# JAX combine for full TVLQR element: (A, c, C, p, P)
# elem has shape (3n+2, n) with row-block layout:
#   A: rows [0:n]
#   c: row  [n]
#   C: rows [n+1 : 2n+1]
#   p: row  [2n+1]
#   P: rows [-n:]
# -----------------------------
def fn_full(next_elem, prev_elem):
    n = prev_elem.shape[-1]

    A_l = prev_elem[0:n, :]                  # (n,n)
    c_l = prev_elem[n, :]                    # (n,)
    C_l = prev_elem[n + 1 : 2 * n + 1, :]    # (n,n)
    p_l = prev_elem[2 * n + 1, :]            # (n,)
    P_l = prev_elem[-n:, :]                  # (n,n)

    A_r = next_elem[0:n, :]
    c_r = next_elem[n, :]
    C_r = next_elem[n + 1 : 2 * n + 1, :]
    p_r = next_elem[2 * n + 1, :]
    P_r = next_elem[-n:, :]

    Ar = A_r @ jnp.linalg.inv(jnp.eye(n, dtype=prev_elem.dtype) + C_l @ P_r)
    Al = A_l.T @ jnp.linalg.inv(jnp.eye(n, dtype=prev_elem.dtype) + P_r @ C_l)

    A_new = Ar @ A_l
    c_new = Ar @ (c_l - C_l @ p_r) + c_r
    C_new = Ar @ C_l @ A_r.T + C_r
    p_new = Al @ (p_r + P_r @ c_l) + p_l
    P_new = Al @ P_r @ A_l + P_l

    return jnp.concatenate(
        [
            A_new,                    # (n,n)
            c_new.reshape(1, n),      # (1,n)
            C_new,                    # (n,n)
            p_new.reshape(1, n),      # (1,n)
            P_new,                    # (n,n)
        ],
        axis=0,
    )


# -----------------------------
# NumPy: associative scan over (A,C,P) with per-level caches
# elems_acp has shape (T, 3n, n) with row-block layout:
#   A: rows [0:n]
#   C: rows [n:2n]
#   P: rows [2n:3n]
#
# caches are (2L, T, n, n) storing values at indices updated at each level.
# levels 0..L-1: upsweep, levels L..2L-1: downsweep
# -----------------------------
def associative_scan_cache_acp(elems_acp, T, n, reverse=False):
    dtype = elems_acp.dtype
    I = np.eye(n, dtype=dtype)
    L = int(np.ceil(np.log2(max(T, 1))))

    cache_Ar  = np.zeros((2 * L, T, n, n), dtype=dtype)
    cache_Al  = np.zeros((2 * L, T, n, n), dtype=dtype)
    cache_ArC = np.zeros((2 * L, T, n, n), dtype=dtype)  # Ar @ C_l
    cache_AlP = np.zeros((2 * L, T, n, n), dtype=dtype)  # Al @ P_r
    cache_Cn  = np.zeros((2 * L, T, n, n), dtype=dtype)
    cache_Pn  = np.zeros((2 * L, T, n, n), dtype=dtype)

    def combine(next_block, prev_block):
        # next_block, prev_block: (k, 3n, n) or (3n, n)
        if prev_block.ndim == 2:
            prev_block = prev_block[None, ...]
            next_block = next_block[None, ...]
            squeeze = True
        else:
            squeeze = False

        A_l = prev_block[:, 0:n, :]
        C_l = prev_block[:, n:2*n, :]
        P_l = prev_block[:, 2*n:3*n, :]

        A_r = next_block[:, 0:n, :]
        C_r = next_block[:, n:2*n, :]
        P_r = next_block[:, 2*n:3*n, :]

        inv1 = np.linalg.inv(I[None, :, :] + np.einsum("bij,bjk->bik", C_l, P_r))
        inv2 = np.linalg.inv(I[None, :, :] + np.einsum("bij,bjk->bik", P_r, C_l))

        Ar = np.einsum("bij,bjk->bik", A_r, inv1)                           # A_r (I + C_l P_r)^-1
        Al = np.einsum("bij,bjk->bik", np.transpose(A_l, (0, 2, 1)), inv2)  # A_l^T (I + P_r C_l)^-1

        ArC = np.einsum("bij,bjk->bik", Ar, C_l)  # Ar @ C_l
        AlP = np.einsum("bij,bjk->bik", Al, P_r)  # Al @ P_r

        A_new = np.einsum("bij,bjk->bik", Ar, A_l)
        C_new = np.einsum("bij,bjk->bik", ArC, np.transpose(A_r, (0, 2, 1))) + C_r
        P_new = np.einsum("bij,bjk->bik", AlP, A_l) + P_l

        out = np.concatenate([A_new, C_new, P_new], axis=1)  # (k, 3n, n)

        if squeeze:
            return out[0], Ar[0], Al[0], ArC[0], AlP[0], C_new[0], P_new[0]
        return out, Ar, Al, ArC, AlP, C_new, P_new

    x = elems_acp[::-1] if reverse else elems_acp
    out = x.copy()

    # Upsweep
    step = 1
    level = 0
    while step < T:
        idx = np.arange(2 * step - 1, T, 2 * step)

        out_new, Ar, Al, ArC, AlP, Cn, Pn = combine(out[idx - step], out[idx])
        out[idx] = out_new

        cache_Ar[level, idx]  = Ar
        cache_Al[level, idx]  = Al
        cache_ArC[level, idx] = ArC
        cache_AlP[level, idx] = AlP
        cache_Cn[level, idx]  = Cn
        cache_Pn[level, idx]  = Pn

        step *= 2
        level += 1

    # Downsweep-style fill
    step //= 4
    level2 = 0
    while step >= 1:
        idx = np.arange(3 * step - 1, T, 2 * step)
        lvl = L + level2

        out_new, Ar, Al, ArC, AlP, Cn, Pn = combine(out[idx - step], out[idx])
        out[idx] = out_new

        cache_Ar[lvl, idx]  = Ar
        cache_Al[lvl, idx]  = Al
        cache_ArC[lvl, idx] = ArC
        cache_AlP[lvl, idx] = AlP
        cache_Cn[lvl, idx]  = Cn
        cache_Pn[lvl, idx]  = Pn

        step //= 2
        level2 += 1

    if reverse:
        out = out[::-1]

    cache = {
        "Ar": cache_Ar,
        "Al": cache_Al,
        "ArC": cache_ArC,
        "AlP": cache_AlP,
        "C_new": cache_Cn,
        "P_new": cache_Pn,
    }
    return out, cache


# -----------------------------
# NumPy: use cached factors to propagate c and p to match JAX fn_full
# No inverses here.
# -----------------------------
def associative_scan_use_cache_cp(c, p, cache, T, reverse=False):
    """
    Reproduces JAX fn_full's (c,p) outputs using cached factors from the A,C,P scan.

    IMPORTANT: This matches the same argument-order used by:
        lax.associative_scan(lambda r,l: vmap(fn_full)(r,l), ..., reverse=...)
    where fn_full(next, prev) treats first arg as "right/next" and second as "left/prev".
    """
    Ar  = cache["Ar"]      # (2L, T, n, n)
    Al  = cache["Al"]
    ArC = cache["ArC"]     # Ar @ C_l (cached at combine time)
    AlP = cache["AlP"]     # Al @ P_r (cached at combine time)

    L2 = Ar.shape[0]
    L = L2 // 2

    c_out = c.copy()
    p_out = p.copy()

    if reverse:
        c_out = c_out[::-1]
        p_out = p_out[::-1]

    # Upsweep
    step = 1
    level = 0
    while step < T:
        idx = np.arange(2 * step - 1, T, 2 * step)

        # Match combine(next=idx-step, prev=idx)
        c_l = c_out[idx]         # prev / left block
        p_l = p_out[idx]
        c_r = c_out[idx - step]  # next / right block
        p_r = p_out[idx - step]

        # c_new = Ar*(c_l - C_l*p_r) + c_r == Ar*c_l - (ArC)*p_r + c_r
        c_out[idx] = (
            np.einsum("bij,bj->bi", Ar[level, idx], c_l)
            - np.einsum("bij,bj->bi", ArC[level, idx], p_r)
            + c_r
        )

        # p_new = Al*(p_r + P_r*c_l) + p_l == Al*p_r + (AlP)*c_l + p_l
        p_out[idx] = (
            np.einsum("bij,bj->bi", Al[level, idx], p_r)
            + np.einsum("bij,bj->bi", AlP[level, idx], c_l)
            + p_l
        )

        step *= 2
        level += 1

    # Downsweep
    step //= 4
    level2 = 0
    while step >= 1:
        idx = np.arange(3 * step - 1, T, 2 * step)
        lvl = L + level2

        c_l = c_out[idx]
        p_l = p_out[idx]
        c_r = c_out[idx - step]
        p_r = p_out[idx - step]

        c_out[idx] = (
            np.einsum("bij,bj->bi", Ar[lvl, idx], c_l)
            - np.einsum("bij,bj->bi", ArC[lvl, idx], p_r)
            + c_r
        )

        p_out[idx] = (
            np.einsum("bij,bj->bi", Al[lvl, idx], p_r)
            + np.einsum("bij,bj->bi", AlP[lvl, idx], c_l)
            + p_l
        )

        step //= 2
        level2 += 1

    if reverse:
        c_out = c_out[::-1]
        p_out = p_out[::-1]

    return c_out, p_out


# -----------------------------
# Dummy data generator
# -----------------------------
def make_dummy_data(T, n, seed=0, scale=0.1, dtype=np.float64):
    rng = np.random.default_rng(seed)

    elems_acp = np.zeros((T, 3 * n, n), dtype=dtype)
    tvlqr_elems = np.zeros((T, 3 * n + 2, n), dtype=dtype)

    c0 = np.zeros((T, n), dtype=dtype)
    p0 = np.zeros((T, n), dtype=dtype)

    for t in range(T):
        A = scale * rng.standard_normal((n, n)).astype(dtype)

        Xc = scale * rng.standard_normal((n, n)).astype(dtype)
        Xp = scale * rng.standard_normal((n, n)).astype(dtype)
        C = (Xc @ Xc.T).astype(dtype)  # PSD-ish
        P = (Xp @ Xp.T).astype(dtype)  # PSD-ish

        c = (scale * rng.standard_normal((n,))).astype(dtype)
        p = (scale * rng.standard_normal((n,))).astype(dtype)

        c0[t] = c
        p0[t] = p

        # NumPy packed (A,C,P)
        elems_acp[t, 0:n, :] = A
        elems_acp[t, n:2*n, :] = C
        elems_acp[t, 2*n:3*n, :] = P

        # JAX packed (A,c,C,p,P)
        tvlqr_elems[t, 0:n, :] = A
        tvlqr_elems[t, n, :] = c
        tvlqr_elems[t, n + 1 : 2 * n + 1, :] = C
        tvlqr_elems[t, 2 * n + 1, :] = p
        tvlqr_elems[t, -n:, :] = P

    return elems_acp, c0, p0, tvlqr_elems


# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    T = 8
    n = 4
    reverse = True
    atol = 1e-6
    rtol = 1e-6

    elems_acp, c0, p0, tvlqr_elems = make_dummy_data(T, n, seed=16, scale=0.1, dtype=np.float64)

    # NumPy: cache structural scan
    rs, cache = associative_scan_cache_acp(elems_acp, T, n, reverse=True)
    # NumPy: propagate c,p using cache
    c_np, p_np = associative_scan_use_cache_cp(c0, p0, cache, T, reverse=True)

    # JAX: full scan
    result = jax.lax.associative_scan(lambda r, l: jax.vmap(fn_full)(r, l), jnp.asarray(tvlqr_elems), reverse=True)
    result_np = np.asarray(result)

    c_jax = result_np[:, n, :]
    p_jax = result_np[:, 2*n + 1, :]
    print("c allclose:", np.allclose(c_np, c_jax, atol=1e-6, rtol=1e-6),
        "max|diff|:", np.max(np.abs(c_np - c_jax)))
    print("p allclose:", np.allclose(p_np, p_jax, atol=1e-6, rtol=1e-6),
        "max|diff|:", np.max(np.abs(p_np - p_jax)))

