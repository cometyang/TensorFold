# Native FP4 indexer on DGX Spark

This opt-in path executes DeepSeek V4.1 Flash indexer QK products with SM121 native FP4 MMA and FP32 accumulation. It is based on the deployed TP2 revision `629c524` plus the recipe patches on `codex/spark-tp2-baseline`.

The deployed indexer already stores keys as E2M1 FP4 with UE8M0 scales per 32 elements. Its scoring kernels expand those values to BF16. This implementation packs the already FP4-rounded queries and uses native block-scaled FP4 instructions. Model weights retain their existing EXL3 format.

Four adjacent query rows can reuse a key tile. Different pool bases trigger separate loads; partial groups use the original logical row count. The original 8192-byte query allocation and 68-byte key representation remain unchanged. Packed queries occupy 2176 useful bytes within the existing query stride, so this change does not reduce allocated query memory.

The TREE prefill variants keep the key MMA fragments in registers across those rows, load packed keys in 16-byte vectors, and reduce head scores with warp shuffles. The shuffle tree preserves the existing FP32 multiply, FMA, and addition order. This removes the shared-memory score matrix and avoids reloading key fragments for each row. Fully invisible tiles write the same masked output without evaluating QK. Decode, candidate-pool kernels, cache representation, and model weights are unchanged.

The CUDA MMA and fragment-loading patterns are adapted from DeepGEMM at `b64107f2b9599ca76445b7f62eedf66bae1d095b`; see `LICENSES/DeepGEMM-MIT.txt` and the family attribution. The implementation preserves the original prompt/decode FP32 FMA and reduction orders.

## Build

Use the deployed CUDA 13.3 / Triton 3.7.1 environment, Zig 0.17.0, and the exact original AOT kit. Compilation runs ahead of time; Python is not part of served inference.

```sh
zig build dsv41-serve -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=fast -j2
python3 tools/dsv41-zig/fp4_index/build.py \
  --base /path/to/original/aot \
  --base-runtime /path/to/original/aot \
  --out /path/to/native-fp4
```

`--base-runtime` is the original kit's absolute path inside the serving environment. Unchanged cubins are symlinked there. For a self-contained package, copy each catalog-referenced cubin into the new catalog directory, resolving those links against the original kit.

The compiler rejects missing native FP4 MMA and BF16 MMA emulation. It replaces the query quantizer and every variant of all three scoring functions as one set. The runtime checks the geometry, stride, baseline-manifest SHA256, replacement coverage, and unchanged kernel entries before loading the opt-in catalog.

Set this environment variable on **both ranks**, then use the existing model launcher and settings:

```sh
export TF_DS_INDEX_FP4_AOT=/path/to/native-fp4
```

An unset variable selects the original catalog. The new optional AOT `grid_x_divisor` defaults to one for existing kernels. Grouped native scoring receives the original logical grid width as an additional argument and launches its ceiling-divided grid.

## Validate

```sh
zig build test -j2 --summary all
python3 tools/dsv41-zig/fp4_index/check.py \
  --base /path/to/original/aot --native /path/to/native-fp4 --out check.json
python3 tools/dsv41-zig/fp4_index/check.py \
  --base /path/to/original/aot --native /path/to/native-fp4 \
  --out variants.json --all-variants
python3 tools/dsv41-zig/fp4_index/check.py \
  --base /path/to/original/aot --native /path/to/native-fp4 \
  --out shared-base.json --all-variants --same-base
python3 tools/dsv41-zig/fp4_index/check.py \
  --base /path/to/original/aot --native /path/to/native-fp4 \
  --out prefill-partial.json --all-variants --prefill-only --prefill-rows 17
python3 tools/dsv41-zig/fp4_index/prefill_bench.py \
  --baseline /path/to/previous/native-fp4 --candidate /path/to/native-fp4 \
  --out prefill-speed.json
```

The default numerical gate covers 2048, 65536, and 1048320 compressed keys. The served model compresses its history, so this kernel-level key count is distinct from prompt length. Additional options cover one-row decode, five/six-row drafted decode, 2048-row prompt chunks, and shared/mixed pool bases.

Query tests compare reconstructed BF16 bits, including zero signs and rounding boundaries. Scoring tests compare every output score/key and tile maximum with the original cubins, and compare sampled columns with an independent FP64 reference. Internal CUDA-event timings are exploratory and include Python launch overhead; served A/B/A measurements determine model performance.

The dedicated prefill benchmark checks byte-exact scores/keys and tile maxima, including a partial row group with tied zero scores and a tail key tile. It validates CUDA-graph replay against uncaptured output, then alternates baseline/candidate timing order for seven sample pairs. Its scanned-key counts are kernel shapes, not whole-model context lengths. The full-model comparison must retain the same clock cap on both ranks and report cold prefill separately from decode.

The benchmark receipt under `docs/receipts/dsv41-native-fp4-sm121-20261028/` records the two-Spark speed and accuracy comparison, clocks, exact returned token IDs, and limitations. The near-1M prompt uses 1048320 tokens to leave generation and pool-verification reserve inside the 1048576-position window.
