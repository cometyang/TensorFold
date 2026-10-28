// Native SM12x scoring with DeepGEMM (MIT) MMA/loading patterns; see LICENSES/DeepGEMM-MIT.txt.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

namespace tf_index_fp4 {

__device__ __forceinline__ void matrix4(uint32_t (&d)[4], const void* ptr) {
    const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(address) : "memory");
}

__device__ __forceinline__ void matrix2(uint32_t (&d)[2], const void* ptr) {
    const uint32_t address = uint32_t(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(address) : "memory");
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                    uint16_t sa, uint16_t sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(uint32_t(sa)), "n"(uint16_t(0)), "n"(uint16_t(0)),
          "r"(uint32_t(sb)), "n"(uint16_t(0)), "n"(uint16_t(0)));
}

struct Shared {
    alignas(16) uint8_t q[32][64];
    alignas(16) uint8_t k[64][64];
    uint32_t qs[32], ks[64];
    float weights[32];
    float terms[64][33];
    int64_t position[64];
    int64_t ordered[64];
    int valid[64];
};

__device__ __forceinline__ int64_t key(float score, int64_t position) {
    const int32_t bits = score == 0.0f ? 0 : __float_as_int(score);
    const int32_t ordered = bits < 0 ? bits ^ 0x7fffffff : bits;
    return int64_t((uint64_t(uint32_t(ordered)) << 32) | uint32_t(0xffffffffull - position));
}

template<bool Tree>
__device__ __forceinline__ float reduce(const float* values, const float* weights) {
    if constexpr (!Tree) {
        float x[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = __fmaf_rn(values[i], weights[i], __fmul_rn(values[i + 8], weights[i + 8]));
            x[i] = __fmaf_rn(values[i + 16], weights[i + 16], x[i]);
            x[i] = __fmaf_rn(values[i + 24], weights[i + 24], x[i]);
        }
        #pragma unroll
        for (int step = 4; step > 0; step /= 2) {
            #pragma unroll
            for (int i = 0; i < step; ++i) x[i] = __fadd_rn(x[i], x[i + step]);
        }
        return x[0];
    }
    float half[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        float x[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int at = h * 16 + i;
            x[i] = __fmaf_rn(values[at], weights[at], __fmul_rn(values[at + 8], weights[at + 8]));
        }
        #pragma unroll
        for (int step = 4; step > 0; step /= 2) {
            #pragma unroll
            for (int i = 0; i < step; ++i) x[i] = __fadd_rn(x[i], x[i + step]);
        }
        half[h] = x[0];
    }
    return __fadd_rn(half[0], half[1]);
}

template<bool Base, bool Mask, bool Keys, bool Maxima, bool Skip, bool Dead, bool Candidates, int CB, bool Tree>
__device__ void score(Shared& sm, const __nv_bfloat16* query, const uint8_t* codes, const uint8_t* scales,
                      const __nv_bfloat16* weights, const int64_t* visible, void* output, int n,
                      const int64_t* bases, const uint8_t* mask, int mask_stride, int64_t* maxima, int nt,
                      const int32_t* blocks, int block_stride, int nout, int r, bool reload, int64_t key_vis) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int b = blockIdx.y;
    const int64_t vis = visible[r];
    const bool alive = !Skip || b * 64 < vis;
    const bool work = !Skip || b * 64 < key_vis;
    const int width = Candidates ? nout : n;
    const auto q = reinterpret_cast<const uint8_t*>(query) + int64_t(r) * 8192;
    if (work) {
        for (int i = tid; i < 128; i += 128) {
            const int row = i / 4, col = (i % 4) * 16;
            *reinterpret_cast<uint4*>(&sm.q[row][col ^ ((row & 6) << 3)]) =
                *reinterpret_cast<const uint4*>(q + row * 64 + col);
        }
        if (tid < 32) {
            sm.qs[tid] = reinterpret_cast<const uint32_t*>(q + 2048)[tid];
            sm.weights[tid] = __bfloat162float(weights[int64_t(r) * 32 + tid]);
        }
        if (reload && tid < 64) {
            const int j = b * 64 + tid;
            int64_t t = j;
            bool valid = j < width;
            if constexpr (Candidates) {
                const int64_t block = valid ? blocks[int64_t(r) * block_stride + j / CB] : -1;
                t = block * CB + j % CB;
                valid = valid && block >= 0 && t < n;
            }
            sm.position[tid] = t;
            sm.valid[tid] = valid;
            const bool load = valid && (!Base || t < key_vis);
            const int64_t physical = t + (Base ? bases[r] : 0);
            uint32_t sf = 0x7f7f7f7f;
            if (load) {
                sf = uint32_t(scales[physical * 4]) | (uint32_t(scales[physical * 4 + 1]) << 8)
                   | (uint32_t(scales[physical * 4 + 2]) << 16) | (uint32_t(scales[physical * 4 + 3]) << 24);
            }
            sm.ks[tid] = sf;
        }
        __syncthreads();
        for (int i = tid; reload && i < 1024; i += 128) {
            const int row = i / 16, part = i % 16;
            const int64_t t = sm.position[row];
            const int64_t physical = t + (Base ? bases[r] : 0);
            const bool load = sm.valid[row] && (!Base || t < key_vis);
            const uint32_t v = load ? *reinterpret_cast<const uint32_t*>(codes + physical * 64 + part * 4) : 0;
            const uint16_t low = (v & 15) | ((v >> 4) & 240) | ((v >> 8) & 3840) | ((v >> 12) & 61440);
            const uint16_t high = ((v >> 4) & 15) | ((v >> 8) & 240) | ((v >> 12) & 3840) | ((v >> 16) & 61440);
            const int swizzle = (row & 6) << 3;
            *reinterpret_cast<uint16_t*>(&sm.k[row][(part * 2) ^ swizzle]) = low;
            *reinterpret_cast<uint16_t*>(&sm.k[row][(part * 2 + 32) ^ swizzle]) = high;
        }
        __syncthreads();
        const int g = lane / 4, t = lane % 4;
        const int arow = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const uint32_t sf_a = sm.ks[warp * 16 + g + (t & 1) * 8];
        uint32_t a0[4], a1[4];
        matrix4(a0, &sm.k[arow][((lane >> 4) * 16) ^ ((arow & 6) << 3)]);
        matrix4(a1, &sm.k[arow][((lane >> 4) * 16 + 32) ^ ((arow & 6) << 3)]);
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int brow = tile * 8 + (lane & 7);
            uint32_t b0[2], b1[2];
            matrix2(b0, &sm.q[brow][(((lane >> 3) & 1) * 16) ^ ((brow & 6) << 3)]);
            matrix2(b1, &sm.q[brow][(((lane >> 3) & 1) * 16 + 32) ^ ((brow & 6) << 3)]);
            const uint32_t sf_b = sm.qs[tile * 8 + g];
            float low[4] = {}, high[4] = {};
            mma(low, a0, b0, uint16_t(sf_a), uint16_t(sf_b));
            mma(high, a1, b1, uint16_t(sf_a >> 16), uint16_t(sf_b >> 16));
            const int h = tile * 8 + t * 2;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = warp * 16 + g + (i / 2) * 8;
                const int head = h + i % 2;
                sm.terms[row][head] = fmaxf(__fadd_rn(low[i], high[i]), 0.0f);
            }
        }
        __syncthreads();
    }
    if (tid < 64) {
        const int j = b * 64 + tid;
        const int64_t position = work ? sm.position[tid] : j;
        float value = work ? reduce<Tree>(sm.terms[tid], sm.weights) : -CUDART_INF_F;
        if (position >= vis) value = -CUDART_INF_F;
        if constexpr (Mask) {
            if (j < n && mask[int64_t(r) * mask_stride + j / CB] == 0) value = -CUDART_INF_F;
        }
        int64_t ordered = key(value, position);
        if constexpr (Candidates) if (!sm.valid[tid]) ordered = INT64_MIN;
        if (j < width && (!Dead || alive)) {
            if constexpr (Keys) reinterpret_cast<int64_t*>(output)[int64_t(r) * width + j] = ordered;
            else reinterpret_cast<float*>(output)[int64_t(r) * width + j] = value;
        }
        if constexpr (Maxima) {
            sm.ordered[tid] = j < width && (!Dead || alive) ? ordered : INT64_MIN;
        }
    }
    __syncthreads();
    if constexpr (Maxima) {
        if (tid == 0) {
            int64_t maximum = INT64_MIN;
            #pragma unroll
            for (int i = 0; i < 64; ++i) maximum = sm.ordered[i] > maximum ? sm.ordered[i] : maximum;
            maxima[int64_t(r) * nt + b] = maximum;
        }
    }
    __syncthreads();
}

}  // namespace tf_index_fp4
