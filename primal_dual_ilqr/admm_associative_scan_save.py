# Need to cache
# ArIClPr_inv
# AlTIPrCl_inv
# C_new
# P_new

import numpy as np
import jax
import jax.numpy as jnp

import numpy as np

def associative_scan_cache(elems, T, n, reverse=False, dtype=None):
    """
    Inclusive associative scan over axis 0 with per-level caches.

    elems: (T, 3n, n) where blocks are [A; C; P]
    returns:
      out: (T, 3n, n)
      cache: dict of arrays with shape (2L, T, n, n)
             levels 0..L-1   = upsweep levels
             levels L..2L-1  = downsweep levels
    """
    dtype = elems.dtype

    I = np.eye(n, dtype=dtype)

    # Number of tree levels (ceil log2)
    L = int(np.ceil(np.log2(max(T, 1))))

    # Allocate caches: (2L, T, n, n)
    cache_Ar = np.zeros((2 * L, T, n, n), dtype=dtype)
    cache_Al = np.zeros((2 * L, T, n, n), dtype=dtype)
    cache_Cn = np.zeros((2 * L, T, n, n), dtype=dtype)
    cache_Pn = np.zeros((2 * L, T, n, n), dtype=dtype)

    def combine(next, prev):
        # next, prev: (k, 3n, n) or (3n, n)
        if prev.ndim == 2:
            prev = prev[None, ...]
            next = next[None, ...]
            squeeze = True
        else:
            squeeze = False

        A_l = prev[:, 0:n, :]
        C_l = prev[:, n:2*n, :]
        P_l = prev[:, 2*n:3*n, :]

        A_r = next[:, 0:n, :]
        C_r = next[:, n:2*n, :]
        P_r = next[:, 2*n:3*n, :]

        inv1 = np.linalg.inv(I[None, :, :] + np.einsum('bij,bjk->bik', C_l, P_r))
        inv2 = np.linalg.inv(I[None, :, :] + np.einsum('bij,bjk->bik', P_r, C_l))

        Ar = np.einsum('bij,bjk->bik', A_r, inv1)
        Al = np.einsum('bij,bjk->bik', np.transpose(A_l, (0, 2, 1)), inv2)

        A_new = np.einsum('bij,bjk->bik', Ar, A_l)
        C_new = (
            np.einsum(
                'bij,bjk->bik',
                np.einsum('bij,bjk->bik', Ar, C_l),
                np.transpose(A_r, (0, 2, 1)),
            )
            + C_r
        )
        P_new = (
            np.einsum(
                'bij,bjk->bik',
                np.einsum('bij,bjk->bik', Al, P_r),
                A_l,
            )
            + P_l
        )

        out = np.concatenate([A_new, C_new, P_new], axis=1)  # (k, 3n, n)

        if squeeze:
            return out[0], Ar[0], Al[0], C_new[0], P_new[0]
        return out, Ar, Al, C_new, P_new

    # Reverse handling
    x = elems[::-1] if reverse else elems
    out = x.copy()

    # --------------------
    # Upsweep (levels 0..L-1)
    # --------------------
    step = 1
    level = 0
    while step < T:
        idx = np.arange(2 * step - 1, T, 2 * step)

        out_new, Ar, Al, Cn, Pn = combine(out[idx - step], out[idx])
        out[idx] = out_new

        cache_Ar[level, idx] = Ar
        cache_Al[level, idx] = Al
        cache_Cn[level, idx] = Cn
        cache_Pn[level, idx] = Pn

        step *= 2
        level += 1

    # --------------------
    # Downsweep-style fill (levels L..2L-1)
    # --------------------
    step //= 4
    level2 = 0
    while step >= 1:
        idx = np.arange(3 * step - 1, T, 2 * step)

        out_new, Ar, Al, Cn, Pn = combine(out[idx - step], out[idx])
        out[idx] = out_new

        cache_Ar[L + level2, idx] = Ar
        cache_Al[L + level2, idx] = Al
        cache_Cn[L + level2, idx] = Cn
        cache_Pn[L + level2, idx] = Pn

        step //= 2
        level2 += 1

    if reverse:
        out = out[::-1]
        cache_Ar = cache_Ar[:, ::-1]
        cache_Al = cache_Al[:, ::-1]
        cache_Cn = cache_Cn[:, ::-1]
        cache_Pn = cache_Pn[:, ::-1]

    cache = {
        "ArIClPr_inv": cache_Ar,   # (2L, T, n, n)
        "AlTIPrCl_inv": cache_Al,  # (2L, T, n, n)
        "C_new": cache_Cn,         # (2L, T, n, n)
        "P_new": cache_Pn,         # (2L, T, n, n)
    }
    return out, cache

def associative_scan_use_cache(
    p, b,
    cache,
    T, n,
    reverse=False
):
    """
    Uses cached scan factors to propagate p and b.

    Args:
        p: (T, n)     initial p_k,k
        b: (T, n)     initial b_k,k
        cache: dict from associative_scan_cache_levels
        T, n: sizes
        reverse: must match the original scan

    Returns:
        p_out: (T, n)   p_{i,j}
        b_out: (T, n)   b_{i,j}
    """

    Ar = cache["ArIClPr_inv"]   # (2L, T, n, n)
    Al = cache["AlTIPrCl_inv"]
    Cn = cache["C_new"]
    Pn = cache["P_new"]

    L2 = Ar.shape[0]
    L = L2 // 2

    p_out = p.copy()
    b_out = b.copy()

    if reverse:
        p_out = p_out[::-1]
        b_out = b_out[::-1]

    # --------------------
    # Upsweep
    # --------------------
    step = 1
    level = 0
    while step < T:
        idx = np.arange(2 * step - 1, T, 2 * step)

        # k = idx, i = idx - step
        pkj = p_out[idx]
        pik = p_out[idx - step]

        bik = b_out[idx - step]
        bkj = b_out[idx]

        Pkj = Pn[level, idx]
        Cik = Cn[level, idx]

        # p update
        p_out[idx] = (
            np.einsum('bij,bj->bi', Al[level, idx], pkj - np.einsum('bij,bj->bi', Pkj, bik))
            + pik
        )

        # b update
        b_out[idx] = (
            np.einsum('bij,bj->bi', Ar[level, idx], bik - np.einsum('bij,bj->bi', Cik, pkj))
            + bkj
        )

        step *= 2
        level += 1

    # --------------------
    # Downsweep
    # --------------------
    step //= 4
    level2 = 0
    while step >= 1:
        idx = np.arange(3 * step - 1, T, 2 * step)
        lvl = L + level2

        pkj = p_out[idx]
        pik = p_out[idx - step]

        bik = b_out[idx - step]
        bkj = b_out[idx]

        Pkj = Pn[lvl, idx]
        Cik = Cn[lvl, idx]

        p_out[idx] = (
            np.einsum('bij,bj->bi', Al[lvl, idx], pkj - np.einsum('bij,bj->bi', Pkj, bik))
            + pik
        )

        b_out[idx] = (
            np.einsum('bij,bj->bi', Ar[lvl, idx], bik - np.einsum('bij,bj->bi', Cik, pkj))
            + bkj
        )

        step //= 2
        level2 += 1

    if reverse:
        p_out = p_out[::-1]
        b_out = b_out[::-1]

    return p_out, b_out

def fn(next, prev):
    def decompose(elem):
        n = 4 
        return (
            elem[:n],
            elem[n],
            elem[n + 1 : 2 * n + 1],
            elem[2 * n + 1],
            elem[-n:],
        )

    A_l, c_l, C_l, p_l, P_l = decompose(prev)
    A_r, c_r, C_r, p_r, P_r = decompose(next)

    ArIClPr_inv = A_r @ jnp.linalg.inv(jnp.eye(n) + C_l @ P_r)
    AlTIPrCl_inv = A_l.T @ jnp.linalg.inv(jnp.eye(n) + P_r @ C_l)

    A_new = ArIClPr_inv @ A_l
    c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
    C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
    p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
    P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

    return jnp.concatenate(
        [
            A_new,
            c_new.reshape(1, n),
            C_new,
            p_new.reshape(1, n),
            P_new,
        ]
    )

def make_dummy_elems(T, n, seed=0):
    rng = np.random.default_rng(seed)

    elems = np.zeros((T, 3*n, n))
    tvlqr_elems = np.zeros((T, 3*n + 2, n))

    p_all = np.zeros((T, n))
    b_all = np.zeros((T, n))
    for t in range(T):
        # A: arbitrary but small
        A = 0.1 * rng.standard_normal((n, n))

        # C, P: PSD and small so I + C P is invertible
        Xc = 0.1 * rng.standard_normal((n, n))
        Xp = 0.1 * rng.standard_normal((n, n))

        C = Xc @ Xc.T          # PSD
        P = Xp @ Xp.T          # PSD
        p = 0.1 * rng.standard_normal((n))
        b = 0.1 * rng.standard_normal((n))

        p_all[t] = p
        b_all[t] = b

        elems[t, 0:n, :]     = A
        elems[t, n :2*n, :]  = C
        elems[t, 2*n:3*n, :] = P

        tvlqr_elems[t, 0:n, :]     = A
        tvlqr_elems[t, n, :] = b
        tvlqr_elems[t, n + 1:2*n + 1, :]   = C
        tvlqr_elems[t, 2 * n + 1, :] = p
        tvlqr_elems[t, -n:, :] = P

    return elems, p_all, b_all, tvlqr_elems

# -----------------------------
# Example ops + quick sanity checks
# -----------------------------
if __name__ == "__main__":
    T = 8
    n = 4

    elems, p, b, tvlqr_elems = make_dummy_elems(T, n)
    # Reverse sum scan
    rs, cache = associative_scan_cache(elems, T, n, reverse=True)
    p_out, b_out = associative_scan_use_cache(p, b, cache, T, n, reverse=True)

    result = jax.lax.associative_scan(lambda r, l: jax.vmap(fn)(r, l), tvlqr_elems, reverse=True)
    # print("reverse sum scan:", rs)  # suffix sums, inclusive
    # print("jax result:", result)
    p_jax = result[:, 2 * n + 1, :]
    # rs_np = np.asarray(rs)
    # result_np = np.asarray(result)

    # ok = np.allclose(rs_np, result_np, rtol=1e-6, atol=1e-6)
    p_jax_np = np.asarray(p_jax)
    p_out_np = np.asarray(p_out)
    ok = np.allclose(p_jax_np, p_out_np, rtol=1e-6, atol=1e-6)
    print("all within tolerance:", ok)