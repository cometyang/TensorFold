// Prefill-only FP4 scoring with resident key fragments and the original TREE reduction order.
#pragma once
#include "index_fp4.cuh"

namespace tf_index_fp4 {

struct PrefillShared {
    alignas(16) uint8_t q[32][64];
    alignas(16) uint8_t k[64][64];
    uint32_t qs[32], ks[64];
    float weights[32];
    int64_t maximum[4];
};

template<bool Keys, bool Maxima>
__device__ void prefill(PrefillShared& sm, const __nv_bfloat16* query, const uint8_t* codes,
                       const uint8_t* scales, const __nv_bfloat16* weights, const int64_t* visible,
                       void* output, int n, int64_t* maxima, int nt, int first, int end) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tile = blockIdx.y * 64, g = lane >> 2, t = lane & 3;
    int64_t maximum_vis = 0;
    for (int r = first; r < end; ++r) maximum_vis = max(maximum_vis, visible[r]);
    const bool work = tile < maximum_vis;
    uint32_t a0[4], a1[4], sf_a = 0;
    if (work) {
        if (tid < 64) sm.ks[tid] = tile + tid < n ? reinterpret_cast<const uint32_t*>(scales)[tile + tid] :
                                                 0x7f7f7f7f;
        // One vector load per 16-byte fragment; transpose its nibbles into the existing MMA key layout.
        for (int i = tid; i < 256; i += 128) {
            const int row = i / 4, part = (i & 3) * 4;
            uint4 packed = tile + row < n ? reinterpret_cast<const uint4*>(codes)[(tile + row) * 4 + (i & 3)] :
                                           make_uint4(0, 0, 0, 0);
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t v = words[j];
                const uint16_t low = (v & 15) | ((v >> 4) & 240) | ((v >> 8) & 3840) | ((v >> 12) & 61440);
                const uint16_t high = ((v >> 4) & 15) | ((v >> 8) & 240) | ((v >> 12) & 3840) | ((v >> 16) & 61440);
                const int swizzle = (row & 6) << 3, col = (part + j) * 2;
                *reinterpret_cast<uint16_t*>(&sm.k[row][col ^ swizzle]) = low;
                *reinterpret_cast<uint16_t*>(&sm.k[row][(col + 32) ^ swizzle]) = high;
            }
        }
        __syncthreads();
        const int arow = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        sf_a = sm.ks[warp * 16 + g + (t & 1) * 8];
        matrix4(a0, &sm.k[arow][((lane >> 4) * 16) ^ ((arow & 6) << 3)]);
        matrix4(a1, &sm.k[arow][((lane >> 4) * 16 + 32) ^ ((arow & 6) << 3)]);
    }
    for (int r = first; r < end; ++r) {
        float result[2] = {-CUDART_INF_F, -CUDART_INF_F};
        if (work) {
            const auto q = reinterpret_cast<const uint8_t*>(query) + int64_t(r) * 8192;
            const int row = tid / 4, col = (tid & 3) * 16;
            *reinterpret_cast<uint4*>(&sm.q[row][col ^ ((row & 6) << 3)]) =
                *reinterpret_cast<const uint4*>(q + row * 64 + col);
            if (tid < 32) {
                sm.qs[tid] = reinterpret_cast<const uint32_t*>(q + 2048)[tid];
                sm.weights[tid] = __bfloat162float(weights[int64_t(r) * 32 + tid]);
            }
            __syncthreads();
            float half[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float values[2][4];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int head_tile = h * 2 + j, brow = head_tile * 8 + (lane & 7);
                    uint32_t b0[2], b1[2];
                    matrix2(b0, &sm.q[brow][(((lane >> 3) & 1) * 16) ^ ((brow & 6) << 3)]);
                    matrix2(b1, &sm.q[brow][(((lane >> 3) & 1) * 16 + 32) ^ ((brow & 6) << 3)]);
                    const uint32_t sf_b = sm.qs[head_tile * 8 + g];
                    float low[4] = {}, high[4] = {};
                    mma(low, a0, b0, uint16_t(sf_a), uint16_t(sf_b));
                    mma(high, a1, b1, uint16_t(sf_a >> 16), uint16_t(sf_b >> 16));
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) values[j][i] = fmaxf(__fadd_rn(low[i], high[i]), 0.0f);
                }
                float partial[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int head = h * 16 + t * 2 + (i & 1);
                    partial[i] = __fmaf_rn(values[0][i], sm.weights[head],
                                          __fmul_rn(values[1][i], sm.weights[head + 8]));
                    partial[i] = __fadd_rn(partial[i], __shfl_xor_sync(0xffffffff, partial[i], 2));
                    partial[i] = __fadd_rn(partial[i], __shfl_xor_sync(0xffffffff, partial[i], 1));
                }
                half[h][0] = __fadd_rn(partial[0], partial[1]);
                half[h][1] = __fadd_rn(partial[2], partial[3]);
            }
            result[0] = __fadd_rn(half[0][0], half[1][0]);
            result[1] = __fadd_rn(half[0][1], half[1][1]);
        }
        int64_t maximum = INT64_MIN;
        if (t == 0) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int position = tile + warp * 16 + g + i * 8;
                const float value = position < visible[r] ? result[i] : -CUDART_INF_F;
                const int64_t ordered = key(value, position);
                if (position < n) {
                    if constexpr (Keys) reinterpret_cast<int64_t*>(output)[int64_t(r) * n + position] = ordered;
                    else reinterpret_cast<float*>(output)[int64_t(r) * n + position] = value;
                    maximum = max(maximum, ordered);
                }
            }
        }
        if constexpr (Maxima) {
            #pragma unroll
            for (int distance = 16; distance >= 4; distance /= 2) {
                const int64_t other = __shfl_xor_sync(0xffffffff, maximum, distance);
                maximum = max(maximum, other);
            }
            if (lane == 0) sm.maximum[warp] = maximum;
            __syncthreads();
            if (tid == 0) maxima[int64_t(r) * nt + blockIdx.y] =
                max(max(sm.maximum[0], sm.maximum[1]), max(sm.maximum[2], sm.maximum[3]));
        }
        __syncthreads();
    }
}

}  // namespace tf_index_fp4
