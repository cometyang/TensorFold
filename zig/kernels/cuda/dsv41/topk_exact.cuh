#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Full-precision score keys and warp collectives shared by the exact selection kernels.
namespace tf_topk_exact {
using Key = unsigned long long;
__device__ __forceinline__ Key warpMin(Key x) {
    for (int d = 16; d; d >>= 1) x = min(x, __shfl_xor_sync(0xffffffffu, x, d));
    return x;
}

__device__ __forceinline__ Key warpMax(Key x) {
    for (int d = 16; d; d >>= 1) x = max(x, __shfl_xor_sync(0xffffffffu, x, d));
    return x;
}

// A single shared atomic reserves a contiguous output span for the selected lanes of a warp.
__device__ __forceinline__ int reserve(bool keep, int* count) {
    const unsigned int active = __ballot_sync(0xffffffffu, keep);
    const int lane = threadIdx.x & 31;
    int first = 0;
    if (active && lane == __ffs(active) - 1) first = atomicAdd(count, __popc(active));
    if (active) first = __shfl_sync(0xffffffffu, first, __ffs(active) - 1);
    return first + __popc(active & ((1u << lane) - 1));
}

struct IntegerInput {
    const long long* row;
    __device__ Key get(int i) const { return Key(row[i]) ^ 0x8000000000000000ull; }
};

struct ScoreInput {
    const float* row;
    __device__ Key get(int i) const {
        unsigned int bits = __float_as_uint(row[i]);
        if (bits == 0x80000000u) bits = 0;
        const int signed_bits = int(bits);
        const int ordered = signed_bits < 0 ? signed_bits ^ 0x7fffffffu : signed_bits;
        return ((Key(uint32_t(ordered)) << 32) | (0xffffffffu - uint32_t(i))) ^ 0x8000000000000000ull;
    }
};

} // namespace tf_topk_exact
