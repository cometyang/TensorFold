# Exact top-k on SM121

Set `TF_DS_TOPK_EXACT=1` on both tensor-parallel ranks to enable the experimental selector.
Unset it to retain the original selection kernels. FP4 scoring is controlled independently by
`TF_DS_INDEX_FP4_AOT`. Existing short-row fused selection and exact tile-max pruning remain in place.

The selector preserves the signed 64-bit score/index ordering, lower-index tie preference,
negative-zero normalization, visibility masking, and ascending output indices. It does not
round scores or change indexer dot products. Integer top-k returns the same multiset of values;
the existing finish kernel establishes the required index order.

Each bounded tile loads its keys into registers, skips common leading bytes, uses warp-aggregated
histograms, and stops radix refinement once the selected prefix contains exactly the required
count. Greater keys are emitted before boundary duplicates. Large rows use independent local
top-k selections followed by an exact merge. A global winner must be among its tile's local
winners, so this partitioning does not approximate selection.

The dispatch policy selects smaller tiles for up to eight query rows and larger tiles for prefill.
It bounds each tile and merge to 32,768 keys, k to 2,048, input length to 1,048,576, and scratch
to 32 MiB per rank. Unsupported shapes use the original implementation. Scratch is allocated
before graph capture and reused on the engine's serialized execution stream.

This work was motivated by [DeepSelect](https://github.com/deepseek-ai/DeepSelect)'s input-reuse
and small-batch parallelism ideas. An initial streaming-threshold prototype regressed important
shapes on GB10. The retained implementation uses register-tiled radix selection and portable
two-kernel merging; it does not depend on DeepSelect or its BF16 input interface.

Run these commands on a qualified SM121 device with CUDA 13 and both GPUs capped at 2200 MHz:

```sh
python3 tools/dsv41-zig/topk_exact/check.py --out topk-kernels.json
nvcc -arch=sm_121 -std=c++20 -O3 --fmad=false --ftz=false -lineinfo \
  -Izig/kernels/cuda tools/dsv41-zig/topk_exact/sanitize.cu -o topk-sanitize
./topk-sanitize
compute-sanitizer --tool memcheck --error-exitcode 99 ./topk-sanitize quick
compute-sanitizer --tool racecheck --error-exitcode 99 ./topk-sanitize quick
zig build test --summary all
```

The Python gate compares original and candidate GPU kernels against a CPU total-order reference
and times graph replay. Its shapes cover small rows, 16K candidate pools, 32K pruned inputs,
larger source rows, decode batches, and prefill batches. The standalone CUDA gate also covers
duplicate integer extrema, signed zero, infinities, NaN bit patterns, noncontiguous row strides,
and per-row visibility. Device timing is not an end-to-end model speedup claim.
