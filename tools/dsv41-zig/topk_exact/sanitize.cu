#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "dsv41/topk_radix.cuh"

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

long long scoreKey(float score, int position) {
    uint32_t bits;
    std::memcpy(&bits, &score, 4);
    if (bits == 0x80000000u) bits = 0;
    const int32_t signed_bits = int32_t(bits);
    const int32_t ordered = signed_bits < 0 ? signed_bits ^ 0x7fffffffu : signed_bits;
    return (uint64_t(uint32_t(ordered)) << 32) | (0xffffffffu - uint32_t(position));
}

int main(int argc, char**) {
    const int shapes[][2] = {{513, 511}, {16384, 512}, {32768, 512}, {131072, 2048}, {1048320, 512}};
    int cases = 0;
    for (const auto& shape : shapes) {
        const int n = shape[0], k = shape[1], rows = 3, stride = n + 7;
        if (argc > 1 && n > 131072) continue;
        std::vector<long long> keys(rows * stride), visible{n, n / 2, 0};
        std::vector<float> score(rows * stride);
        long long *ki, *out, *vis, *scratch;
        float* sc;
        const int parts = n <= 16384 ? 1 : std::max((n + 32767) / 32768, std::min(16, (n + 4095) / 4096));
        check(cudaMalloc(&ki, keys.size() * 8));
        check(cudaMalloc(&sc, score.size() * 4));
        check(cudaMalloc(&out, rows * k * 8));
        check(cudaMalloc(&vis, rows * 8));
        check(cudaMalloc(&scratch, rows * parts * k * 8));
        check(cudaMemcpy(vis, visible.data(), rows * 8, cudaMemcpyHostToDevice));
        for (int pattern = 0; pattern < 6; ++pattern) {
            for (int row = 0; row < rows; ++row) {
                for (int i = 0; i < n; ++i) {
                    const uint64_t mix = uint64_t(i + 31 * row) * 6364136223846793005ull + 1442695040888963407ull;
                    long long value = mix;
                    if (pattern == 1) value = LLONG_MIN;
                    if (pattern == 2) value = (i % 71 == 0) ? LLONG_MAX : LLONG_MIN;
                    if (pattern == 3) value = i % 3 - 1;
                    if (pattern == 4) value = i;
                    if (pattern == 5) value = -i;
                    keys[row * stride + i] = value;
                    uint32_t bits = uint32_t(mix >> 32);
                    if (pattern == 1) bits = i % 2 ? 0x80000000u : 0;
                    if (pattern == 2) bits = i % 71 ? 0xff800000u : 0x7f800000u;
                    if (pattern == 3) bits = 0x7fc00000u + i % 7;
                    std::memcpy(&score[row * stride + i], &bits, 4);
                    if (pattern == 4) score[row * stride + i] = float(i);
                    if (pattern == 5) score[row * stride + i] = -float(i);
                }
            }
            check(cudaMemcpy(ki, keys.data(), keys.size() * 8, cudaMemcpyHostToDevice));
            check(cudaMemcpy(sc, score.data(), score.size() * 4, cudaMemcpyHostToDevice));
            for (int kind = 0; kind < 2; ++kind) {
                if (parts == 1) {
                    if (kind == 0) tf_ds_topk_i64_radix_kernel<<<rows, 1024>>>(ki, stride, n, k, out);
                    else tf_ds_topk_indices_radix_kernel<<<rows, 1024>>>(sc, stride, n, k, vis, out);
                } else {
                    if (kind == 0) {
                        tf_ds_topk_i64_parts_kernel<<<dim3(rows, parts), 1024>>>(ki, stride, n, k, parts, scratch);
                        tf_ds_topk_i64_radix_kernel<<<rows, 1024>>>(scratch, parts * k, parts * k, k, out);
                    } else {
                        tf_ds_topk_indices_parts_kernel<<<dim3(rows, parts), 1024>>>(sc, stride, n, k, parts, scratch);
                        tf_ds_topk_indices_merge_kernel<<<rows, 1024>>>(scratch, parts * k, parts * k, k, vis, out);
                    }
                }
                check(cudaGetLastError());
                check(cudaDeviceSynchronize());
                std::vector<long long> actual(rows * k);
                check(cudaMemcpy(actual.data(), out, actual.size() * 8, cudaMemcpyDeviceToHost));
                for (int row = 0; row < rows; ++row) {
                    std::vector<long long> expected(n);
                    for (int i = 0; i < n; ++i) expected[i] = kind ? scoreKey(score[row * stride + i], i) : keys[row * stride + i];
                    std::sort(expected.begin(), expected.end(), std::greater<long long>());
                    expected.resize(k);
                    if (kind) {
                        for (auto& value : expected) value = 0xffffffffu - uint32_t(value);
                        std::sort(expected.begin(), expected.end());
                        for (auto& value : expected) if (value >= visible[row]) value = -1;
                    } else {
                        std::sort(actual.begin() + row * k, actual.begin() + (row + 1) * k, std::greater<long long>());
                    }
                    if (!std::equal(expected.begin(), expected.end(), actual.begin() + row * k)) {
                        std::fprintf(stderr, "mismatch n=%d k=%d pattern=%d kind=%d row=%d\n", n, k, pattern, kind, row);
                        return 3;
                    }
                }
                ++cases;
            }
        }
        check(cudaFree(ki));
        check(cudaFree(sc));
        check(cudaFree(out));
        check(cudaFree(vis));
        check(cudaFree(scratch));
    }
    std::printf("{\"passed\":true,\"cases\":%d,\"reference\":\"CPU signed 64-bit total order\"}\n", cases);
}
