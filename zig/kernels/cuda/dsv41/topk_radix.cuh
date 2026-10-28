#pragma once
#include "topk_exact.cuh"
#include <limits.h>

namespace tf_topk_radix {
using tf_topk_exact::Key;
constexpr int threads = 1024;
struct Shared {
    unsigned int hist[256];
    int sums[8];
    Key minima[32], maxima[32], prefix, mask;
    long long cols[2048];
    int digit, want, stop, count;
};

// A bounded tile is read once into registers; only the unresolved radix prefix is examined at each pass.
template<int Items, class Input> __device__ void select(Shared& sm, Input input, int begin, int end, int k) {
    const int tid = threadIdx.x;
    Key values[Items];
    Key lo = ~Key(0), hi = 0;
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = begin + j * threads + tid;
        values[j] = i < end ? input.get(i) : 0;
        if (i < end) {
            lo = min(lo, values[j]);
            hi = max(hi, values[j]);
        }
    }
    lo = tf_topk_exact::warpMin(lo);
    hi = tf_topk_exact::warpMax(hi);
    if ((tid & 31) == 0) {
        sm.minima[tid >> 5] = lo;
        sm.maxima[tid >> 5] = hi;
    }
    __syncthreads();
    if (tid < 32) {
        lo = tf_topk_exact::warpMin(sm.minima[tid]);
        hi = tf_topk_exact::warpMax(sm.maxima[tid]);
    }
    if (tid == 0) {
        sm.digit = lo == hi ? -1 : 7 - __clzll(lo ^ hi) / 8;
        sm.mask = sm.digit == 7 ? 0 : (~Key(0) << (8 * (sm.digit + 1)));
        sm.prefix = lo & sm.mask;
        sm.want = min(k, end - begin);
        sm.count = 0;
        sm.stop = 0;
    }
    __syncthreads();
    for (int d = sm.digit; d >= 0; --d) {
        if (tid < 256) sm.hist[tid] = 0;
        __syncthreads();
        const Key prefix = sm.prefix, mask = sm.mask;
        const int wanted = sm.want;
        #pragma unroll
        for (int j = 0; j < Items; ++j) {
            const bool live = begin + j * threads + tid < end && (values[j] & mask) == prefix;
            const unsigned int active = __ballot_sync(0xffffffffu, live);
            if (live) {
                const unsigned int digit = (values[j] >> (8 * d)) & 255;
                const unsigned int same = __match_any_sync(active, digit);
                if ((tid & 31) == __ffs(same) - 1) atomicAdd(sm.hist + digit, __popc(same));
            }
        }
        __syncthreads();
        int inclusive = 0, bin_count = 0;
        if (tid < 256) {
            inclusive = bin_count = sm.hist[255 - tid];
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if ((tid & 31) >= offset) inclusive += other;
            }
            if ((tid & 31) == 31) sm.sums[tid >> 5] = inclusive;
        }
        __syncthreads();
        if (tid < 32) {
            int sum = tid < 8 ? sm.sums[tid] : 0;
            for (int offset = 1; offset < 8; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffffu, sum, offset);
                if (tid >= offset) sum += other;
            }
            if (tid < 8) sm.sums[tid] = sum;
        }
        __syncthreads();
        if (tid < 256) {
            inclusive += tid >= 32 ? sm.sums[(tid >> 5) - 1] : 0;
            if (inclusive >= wanted && inclusive - bin_count < wanted) {
                sm.want = wanted - (inclusive - bin_count);
                sm.prefix |= Key(255 - tid) << (8 * d);
                sm.mask |= Key(255) << (8 * d);
                sm.stop = inclusive == wanted;
            }
        }
        __syncthreads();
        if (sm.stop) break;
    }
    // The caller consumes the tile's winners through cols; ties at the boundary cannot displace greater keys.
    const Key kth = sm.prefix;
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const bool keep = begin + j * threads + tid < end && values[j] > kth;
        const int at = tf_topk_exact::reserve(keep, &sm.count);
        if (keep && at < k) sm.cols[at] = values[j] ^ 0x8000000000000000ull;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const bool equal = begin + j * threads + tid < end && values[j] == kth;
        const int at = tf_topk_exact::reserve(equal, &sm.count);
        if (equal && at < k) sm.cols[at] = values[j] ^ 0x8000000000000000ull;
    }
    for (int j = end - begin + tid; j < k; j += threads) sm.cols[j] = LLONG_MIN;
    __syncthreads();
}

__device__ void indices(Shared& sm, long long visible, long long* out, int k) {
    const int tid = threadIdx.x;
    int count = 1;
    while (count < k) count <<= 1;
    for (int i = tid; i < count; i += threads) sm.cols[i] = i < k ? 0xffffffffu - uint32_t(sm.cols[i]) : LLONG_MAX;
    for (int size = 2; size <= count; size <<= 1) {
        for (int stride = size >> 1; stride; stride >>= 1) {
            __syncthreads();
            for (int i = tid; i < count; i += threads) {
                const int j = i ^ stride;
                if (j > i) {
                    const long long a = sm.cols[i], b = sm.cols[j];
                    if ((a > b) == ((i & size) == 0)) {
                        sm.cols[i] = b;
                        sm.cols[j] = a;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < k; i += threads) out[i] = sm.cols[i] < visible ? sm.cols[i] : -1;
}
} // namespace tf_topk_radix

extern "C" __global__ __launch_bounds__(1024) void tf_ds_topk_i64_radix_kernel(
    const long long* keys, long long ks, int n, int k, long long* out) {
    __shared__ tf_topk_radix::Shared sm;
    if (n <= 16384) tf_topk_radix::select<16>(sm, tf_topk_exact::IntegerInput{keys + blockIdx.x * ks}, 0, n, k);
    else tf_topk_radix::select<32>(sm, tf_topk_exact::IntegerInput{keys + blockIdx.x * ks}, 0, n, k);
    for (int i = threadIdx.x; i < k; i += 1024) out[blockIdx.x * k + i] = sm.cols[i];
}

extern "C" __global__ __launch_bounds__(1024) void tf_ds_topk_indices_radix_kernel(
    const float* score, long long ss, int n, int k, const long long* vis, long long* out) {
    __shared__ tf_topk_radix::Shared sm;
    if (n <= 16384) tf_topk_radix::select<16>(sm, tf_topk_exact::ScoreInput{score + blockIdx.x * ss}, 0, n, k);
    else tf_topk_radix::select<32>(sm, tf_topk_exact::ScoreInput{score + blockIdx.x * ss}, 0, n, k);
    tf_topk_radix::indices(sm, vis[blockIdx.x], out + blockIdx.x * k, k);
}

extern "C" __global__ __launch_bounds__(1024) void tf_ds_topk_i64_parts_kernel(
    const long long* keys, long long ks, int n, int k, int parts, long long* out) {
    __shared__ tf_topk_radix::Shared sm;
    const int width = (n + parts - 1) / parts, begin = blockIdx.y * width, end = min(n, begin + width);
    if (width <= 16384) tf_topk_radix::select<16>(sm, tf_topk_exact::IntegerInput{keys + blockIdx.x * ks}, begin, end, k);
    else tf_topk_radix::select<32>(sm, tf_topk_exact::IntegerInput{keys + blockIdx.x * ks}, begin, end, k);
    for (int i = threadIdx.x; i < k; i += 1024) out[(blockIdx.x * parts + blockIdx.y) * k + i] = sm.cols[i];
}

extern "C" __global__ __launch_bounds__(1024) void tf_ds_topk_indices_parts_kernel(
    const float* score, long long ss, int n, int k, int parts, long long* out) {
    __shared__ tf_topk_radix::Shared sm;
    const int width = (n + parts - 1) / parts, begin = blockIdx.y * width, end = min(n, begin + width);
    if (width <= 16384) tf_topk_radix::select<16>(sm, tf_topk_exact::ScoreInput{score + blockIdx.x * ss}, begin, end, k);
    else tf_topk_radix::select<32>(sm, tf_topk_exact::ScoreInput{score + blockIdx.x * ss}, begin, end, k);
    for (int i = threadIdx.x; i < k; i += 1024) out[(blockIdx.x * parts + blockIdx.y) * k + i] = sm.cols[i];
}

extern "C" __global__ __launch_bounds__(1024) void tf_ds_topk_indices_merge_kernel(
    const long long* keys, long long ks, int n, int k, const long long* vis, long long* out) {
    __shared__ tf_topk_radix::Shared sm;
    if (n <= 16384) tf_topk_radix::select<16>(sm, tf_topk_exact::IntegerInput{keys + blockIdx.x * ks}, 0, n, k);
    else tf_topk_radix::select<32>(sm, tf_topk_exact::IntegerInput{keys + blockIdx.x * ks}, 0, n, k);
    tf_topk_radix::indices(sm, vis[blockIdx.x], out + blockIdx.x * k, k);
}
