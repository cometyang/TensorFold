#!/usr/bin/env bash
set -euo pipefail

source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$source_dir/build"}
cuda_dir=${CUDA_HOME:-/usr/local/cuda-13.3}
target_dir=${CUDA_TARGET_DIR:-"$cuda_dir/targets/sbsa-linux"}
compiler=${CXX:-g++}
mkdir -p -- "$output_dir"

common=(-std=c++17 -O2 -I"$target_dir/include" -L"$target_dir/lib"
        -L"$target_dir/lib/stubs" -Wl,-rpath,"$target_dir/lib")
libraries=(-lcuda -lcupti -pthread)

"$compiler" "${common[@]}" -fPIC -shared "$source_dir/traffic_profile.cpp" \
    "${libraries[@]}" -o "$output_dir/libtf_traffic.so"
"$compiler" "${common[@]}" "$source_dir/traffic_probe.cpp" \
    "$source_dir/traffic_profile.cpp" "${libraries[@]}" -o "$output_dir/traffic_probe"

printf 'Built %s/libtf_traffic.so and %s/traffic_probe\n' "$output_dir" "$output_dir"
