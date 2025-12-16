from __future__ import annotations
from functools import partial
from typing import NamedTuple
import math
import numpy as np
import jax
import jax.numpy as jnp


# -----------------------------
# Cache container (PyTree-friendly)
# -----------------------------
class ACPScanCache(NamedTuple):
    Ar:  jnp.ndarray   # (2L, T, n, n)
    Al:  jnp.ndarray
    ArC: jnp.ndarray
    AlP: jnp.ndarray
    Cn:  jnp.ndarray
    Pn:  jnp.ndarray


# -----------------------------
# JAX combine for full TVLQR element (for reference comparison)
# -----------------------------
def fn_full(next_elem, prev_elem):
    n = prev_elem.shape[-1]

    A_l = prev_elem[0:n, :]
    c_l = prev_elem[n, :]
    C_l = prev_elem[n + 1 : 2 * n + 1, :]
    p_l = prev_elem[2 * n + 1, :]
    P_l = prev_elem[-n:, :]

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
        [A_new, c_new.reshape(1, n), C_new, p_new.reshape(1, n), P_new],
        axis=0,
    )


# -----------------------------
# Helpers (static-shape masking)
# -----------------------------
def _shift_down(x, step):
    pad = jnp.zeros((step,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([pad, x[:-step]], axis=0)

def _mask_upsweep(T, step):
    t = jnp.arange(T)
    period = 2 * step
    return (t % period) == (period - 1)

def _mask_downsweep(T, step):
    t = jnp.arange(T)
    period = 2 * step
    return (t >= (3 * step - 1)) & ((t % period) == (step - 1))

def _masked_write_level(cache, level, vals, mask):
    # cache[level, t] = vals[t] if mask[t] else 0
    m = mask.astype(vals.dtype)[:, None, None]
    return cache.at[level].set(vals * m)


def _combine_acp_all(next_block, prev_block, n):
    """
    next_block, prev_block: (T, 3n, n)
    Returns:
      combined: (T, 3n, n)
      Ar, Al, ArC, AlP, C_new, P_new: (T, n, n)
    """
    dtype = prev_block.dtype
    I = jnp.eye(n, dtype=dtype)[None, :, :]  # (1,n,n) for broadcasting

    A_l = prev_block[:, 0:n, :]
    C_l = prev_block[:, n:2*n, :]
    P_l = prev_block[:, 2*n:3*n, :]

    A_r = next_block[:, 0:n, :]
    C_r = next_block[:, n:2*n, :]
    P_r = next_block[:, 2*n:3*n, :]

    inv1 = jnp.linalg.inv(I + C_l @ P_r)
    inv2 = jnp.linalg.inv(I + P_r @ C_l)

    Ar = A_r @ inv1
    Al = jnp.swapaxes(A_l, -1, -2) @ inv2

    ArC = Ar @ C_l
    AlP = Al @ P_r

    A_new = Ar @ A_l
    C_new = ArC @ jnp.swapaxes(A_r, -1, -2) + C_r
    P_new = AlP @ A_l + P_l

    combined = jnp.concatenate([A_new, C_new, P_new], axis=1)
    return combined, Ar, Al, ArC, AlP, C_new, P_new


# -----------------------------
# JAX: cache-producing scan over (A,C,P)
# -----------------------------
@partial(jax.jit, static_argnums=(1, 2, 3))
def associative_scan_cache_acp_jax(elems_acp, T: int, n: int, reverse: bool = False):
    dtype = elems_acp.dtype
    L = int(math.ceil(math.log2(max(T, 1))))

    Ar  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Al  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    ArC = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    AlP = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Cn  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Pn  = jnp.zeros((2 * L, T, n, n), dtype=dtype)

    x = elems_acp[::-1] if reverse else elems_acp
    out = x

    # Upsweep
    step = 1
    level = 0
    while step < T and level < L:
        mask = _mask_upsweep(T, step)
        next_block = _shift_down(out, step)
        prev_block = out

        combined, aR, aL, aRC, aLP, cN, pN = _combine_acp_all(next_block, prev_block, n)

        m = mask[:, None, None]
        out = jnp.where(m, combined, out)

        Ar  = _masked_write_level(Ar,  level, aR,  mask)
        Al  = _masked_write_level(Al,  level, aL,  mask)
        ArC = _masked_write_level(ArC, level, aRC, mask)
        AlP = _masked_write_level(AlP, level, aLP, mask)
        Cn  = _masked_write_level(Cn,  level, cN,  mask)
        Pn  = _masked_write_level(Pn,  level, pN,  mask)

        step *= 2
        level += 1

    # Downsweep-style fill
    step //= 4
    level2 = 0
    while step >= 1 and level2 < L:
        mask = _mask_downsweep(T, step)
        next_block = _shift_down(out, step)
        prev_block = out

        combined, aR, aL, aRC, aLP, cN, pN = _combine_acp_all(next_block, prev_block, n)

        m = mask[:, None, None]
        out = jnp.where(m, combined, out)

        lvl = L + level2
        Ar  = _masked_write_level(Ar,  lvl, aR,  mask)
        Al  = _masked_write_level(Al,  lvl, aL,  mask)
        ArC = _masked_write_level(ArC, lvl, aRC, mask)
        AlP = _masked_write_level(AlP, lvl, aLP, mask)
        Cn  = _masked_write_level(Cn,  lvl, cN,  mask)
        Pn  = _masked_write_level(Pn,  lvl, pN,  mask)

        step //= 2
        level2 += 1

    if reverse:
        out = out[::-1]

    return out, ACPScanCache(Ar=Ar, Al=Al, ArC=ArC, AlP=AlP, Cn=Cn, Pn=Pn)


# -----------------------------
# JAX: propagate (c,p) using caches (no inverses)
# -----------------------------
@partial(jax.jit, static_argnums=(2, 4))
def associative_scan_use_cache_cp_jax(c, p, T: int, cache: ACPScanCache, reverse: bool = False):
    Ar, Al, ArC, AlP = cache.Ar, cache.Al, cache.ArC, cache.AlP
    L = Ar.shape[0] // 2

    c_out = c[::-1] if reverse else c
    p_out = p[::-1] if reverse else p

    # Upsweep
    step = 1
    level = 0
    while step < T and level < L:
        mask = _mask_upsweep(T, step)
        m = mask.astype(c_out.dtype)[:, None]

        c_l = c_out
        p_l = p_out
        c_r = _shift_down(c_out, step)
        p_r = _shift_down(p_out, step)

        c_new = (Ar[level]  @ c_l[..., None]).squeeze(-1) - (ArC[level] @ p_r[..., None]).squeeze(-1) + c_r
        p_new = (Al[level]  @ p_r[..., None]).squeeze(-1) + (AlP[level] @ c_l[..., None]).squeeze(-1) + p_l

        c_out = c_out + m * (c_new - c_out)
        p_out = p_out + m * (p_new - p_out)

        step *= 2
        level += 1

    # Downsweep
    step //= 4
    level2 = 0
    while step >= 1 and level2 < L:
        mask = _mask_downsweep(T, step)
        m = mask.astype(c_out.dtype)[:, None]
        lvl = L + level2

        c_l = c_out
        p_l = p_out
        c_r = _shift_down(c_out, step)
        p_r = _shift_down(p_out, step)

        c_new = (Ar[lvl]  @ c_l[..., None]).squeeze(-1) - (ArC[lvl] @ p_r[..., None]).squeeze(-1) + c_r
        p_new = (Al[lvl]  @ p_r[..., None]).squeeze(-1) + (AlP[lvl] @ c_l[..., None]).squeeze(-1) + p_l

        c_out = c_out + m * (c_new - c_out)
        p_out = p_out + m * (p_new - p_out)

        step //= 2
        level2 += 1

    if reverse:
        c_out = c_out[::-1]
        p_out = p_out[::-1]

    return c_out, p_out


# -----------------------------
# Dummy data generator
# -----------------------------
def make_dummy_data(T, n, seed=0, scale=0.1, dtype=np.float32):
    rng = np.random.default_rng(seed)

    elems_acp = np.zeros((T, 3 * n, n), dtype=dtype)
    tvlqr_elems = np.zeros((T, 3 * n + 2, n), dtype=dtype)
    c0 = np.zeros((T, n), dtype=dtype)
    p0 = np.zeros((T, n), dtype=dtype)

    for t in range(T):
        A = (scale * rng.standard_normal((n, n))).astype(dtype)
        Xc = (scale * rng.standard_normal((n, n))).astype(dtype)
        Xp = (scale * rng.standard_normal((n, n))).astype(dtype)
        C = (Xc @ Xc.T).astype(dtype)
        P = (Xp @ Xp.T).astype(dtype)
        c = (scale * rng.standard_normal((n,))).astype(dtype)
        p = (scale * rng.standard_normal((n,))).astype(dtype)

        c0[t] = c
        p0[t] = p

        elems_acp[t, 0:n, :] = A
        elems_acp[t, n:2*n, :] = C
        elems_acp[t, 2*n:3*n, :] = P

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
    T = 25
    n = 61
    reverse = True
    atol = 1e-6
    rtol = 1e-6

    elems_acp, c0, p0, tvlqr_elems = make_dummy_data(T, n, seed=1, scale=0.1, dtype=np.float32)

    elems_acp_j = jnp.asarray(elems_acp)
    c0_j = jnp.asarray(c0)
    p0_j = jnp.asarray(p0)
    tvlqr_elems_j = jnp.asarray(tvlqr_elems)

    out_acp, cache = associative_scan_cache_acp_jax(elems_acp_j, T, n, reverse=reverse)
    c_out, p_out = associative_scan_use_cache_cp_jax(c0_j, p0_j, T, cache, reverse=reverse)

    result = jax.lax.associative_scan(lambda r, l: jax.vmap(fn_full)(r, l), tvlqr_elems_j, reverse=reverse)
    

    import time
    start = time.perf_counter()
    out_acp, cache = associative_scan_cache_acp_jax(elems_acp_j, T, n, reverse=reverse)
    c_out, p_out = associative_scan_use_cache_cp_jax(c0_j, p0_j, T, cache, reverse=reverse)
    end = time.perf_counter()
    print(end - start)

    start = time.perf_counter()
    result = jax.lax.associative_scan(lambda r, l: jax.vmap(fn_full)(r, l), tvlqr_elems_j, reverse=reverse)
    end = time.perf_counter()
    print(end - start)

    c_jax = result[:, n, :]
    p_jax = result[:, 2*n + 1, :]

    c_out_np = np.asarray(c_out)
    p_out_np = np.asarray(p_out)
    c_jax_np = np.asarray(c_jax)
    p_jax_np = np.asarray(p_jax)

    print("c allclose:", np.allclose(c_out_np, c_jax_np, atol=atol, rtol=rtol),
          "max|diff|:", np.max(np.abs(c_out_np - c_jax_np)))
    print("p allclose:", np.allclose(p_out_np, p_jax_np, atol=atol, rtol=rtol),
          "max|diff|:", np.max(np.abs(p_out_np - p_jax_np)))
