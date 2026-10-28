"""Opt-in SM121 MXFP4 indexer kernels; retain the existing Zig launch ABI and cache format."""
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@triton.jit
def _round_code(t):
    a = tl.abs(t)
    idx = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
           + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32)
           + (a > 5.0).to(tl.int32))
    idx += ((a == 0.75) | (a == 1.75) | (a == 3.5)).to(tl.int32)
    return idx | ((t.to(tl.int32, bitcast=True) >> 28) & 8)


@triton.jit
def _fp4qd_p2(X, OUT, nblk, BLKS: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    bi = tl.program_id(0) * BLKS + tl.arange(0, BLKS)
    e = tl.arange(0, 32)
    v = tl.load(X + bi[:, None] * 32 + e[None, :], mask=(bi < nblk)[:, None], other=0).to(tl.float32)
    amax = tl.max(tl.abs(v), axis=1)
    sv = tl.div_rn(tl.maximum(amax, 6 * 2.0 ** -126), 6.0)
    bits = sv.to(tl.int32, bitcast=True)
    ex = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (ex << 23).to(tl.float32, bitcast=True)
    values = tl.minimum(tl.maximum(tl.div_rn(v, scale[:, None]), -6.0), 6.0)
    code = _round_code(values)
    pair = tl.reshape(code, (BLKS, 16, 2))
    packed = (tl.sum(tl.where(tl.arange(0, 2)[None, None, :] == 0, pair, pair << 4), axis=2))
    out = OUT.to(tl.pointer_type(tl.uint8))
    row = bi // 128
    head = bi // 4 % 32
    group = bi % 4
    at = row * 8192 + head * 64 + group * 16
    tl.store(out + at[:, None] + tl.arange(0, 16)[None, :], packed.to(tl.uint8), mask=(bi < nblk)[:, None])
    tl.store(out + row * 8192 + 2048 + head * 4 + group, ex.to(tl.uint8), mask=bi < nblk)
