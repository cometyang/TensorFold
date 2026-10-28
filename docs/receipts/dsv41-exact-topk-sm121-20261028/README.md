# Exact radix top-k on two DGX Sparks

The opt-in selector preserves the existing total order and substantially reduces standalone
selection latency. In the full model, near-1M plain decode improved from 40.12 to 41.38 tokens/s
(+3.14% against the mean of the before/after baselines). Speculative decode was essentially
unchanged. Cold-prefill gains are not established because the repeated baseline drifted.
The short-context confirmation found plain decode essentially unchanged at 2K and +1.89% at
64K, with speculative decode changes of -0.03% and +1.21% respectively. Short-context prefill
was essentially unchanged in that repeat.

This comparison isolates **native FP4 plus exact top-k versus native FP4 alone**. It does not
compare with the original production configuration, and its percentages must not be added
to the earlier FP4 receipt. The original production service was restored after testing.

## Full model results

The sequence was baseline A1, candidate B, baseline A2. Each short-context cell is a median
over three saved synthetic fixtures. The near-1M cell has one fixture per phase. All requests
generated 128 tokens. These are descriptive observations, not confidence intervals.

| Prompt tokens | Plain decode A1 | Candidate | Plain decode A2 | Candidate vs mean baseline |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 44.317 | 44.171 | 44.035 | -0.01% |
| 65,536 | 43.372 | 43.929 | 40.900 | +4.25%, confounded by baseline drift |
| 1,048,320 | 40.137 | 41.380 | 40.103 | +3.14% |

Decode rates are tokens/s, measured as `(completion_tokens - 1) / (last_token_time - first_token_time)`.
The baseline summary is `(median(A1) + median(A2)) / 2`; decode change is
`100 * (candidate / baseline - 1)`. Run `python3 summarize.py` in this directory to independently
recompute these values and check the stored token IDs, clock samples, and fixture hashes.

| Prompt tokens | Speculative decode A1 | Candidate | Speculative decode A2 | Candidate vs mean baseline |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 98.015 | 97.406 | 97.907 | -0.57% |
| 65,536 | 85.240 | 86.579 | 83.211 | +2.79%, descriptive only |
| 1,048,320 | 64.997 | 65.006 | 64.403 | +0.47% |

| Prompt tokens | Cold TTFT A1, s | Candidate, s | Cold TTFT A2, s | A2 baseline drift |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 1.136 | 1.124 | 1.148 | +1.04% |
| 65,536 | 28.028 | 27.938 | 32.752 | +16.86% |
| 1,048,320 | 506.377 | 503.678 | 544.518 | +7.53% |

The full-run mean-baseline calculation would suggest 8.07% and 4.14% TTFT reductions at 64K
and near 1M. **Those are not established improvements:** the candidate was only 0.32% and 0.53%
faster than A1. Clocks remained capped, no additional GPU process was found, and diagnostics
showed no active swapping. The cause of the baseline drift remains unresolved.

## Short-context confirmation

An additional A/B/A sweep reused the exact six saved 2K/64K fixtures, the same binaries,
128-token outputs, and the 2200 MHz ceiling. Values below compare the candidate with the mean
of the before/after phase medians. The original full-run observations above remain intact.

| Prompt tokens | Plain decode, tokens/s | Speculative decode, tokens/s | Cold TTFT, seconds |
| --- | ---: | ---: | ---: |
| 2,048 | 44.163 → 44.128 (-0.08%) | 97.286 → 97.256 (-0.03%) | 1.137 → 1.136 (-0.15%) |
| 65,536 | 43.161 → 43.976 (+1.89%) | 84.784 → 85.807 (+1.21%) | 28.064 → 28.036 (-0.10%) |

Baseline A2 versus A1 drift in this repeat:

- 2,048: TTFT +0.13%, plain decode +0.08%, speculative decode -0.77%
- 65,536: TTFT +0.53%, plain decode -0.49%, speculative decode +0.89%

All 32 cross-phase reply comparisons, 24 speculative/plain pairs, six concurrent replies,
and five lifecycle replays matched exactly. All replies also matched the corresponding
full-run A1 token IDs. Retrieval passed 9/9 records at each context in every phase.
The original service was restored. The confirmation evidence is in `confirmation/`.

Clock monitoring captured 112 paired samples, zero errors or observed violations, and maxima of 2190/2190 MHz.


## Correctness and lifecycle

- All 36 cross-phase full-output comparisons matched all 128 returned token IDs and output hashes.
- All 27 serial speculative/plain pairs matched, as did six concurrent replies against their solo references.
- Retrieval passed 9/9 records at 2K, 9/9 at 64K, and 3/3 near 1M in each phase. These are small synthetic tests.
- Five candidate replays matched the reference across cancellation during cold prefill, cancellation during decode, and retained-prefix reuse.
- 240 GPU microbenchmark/correctness cases matched the CPU total-order reference and original kernels.
- 60 standalone CUDA cases passed; Compute Sanitizer reported zero memory errors across 48 cases and zero race hazards across 48 cases.
- 231/231 host tests and the native server/worker build passed.
- The lean checker reports 1,490 inherited findings on both base and candidate, with zero additional findings. It is not globally clean.

No broad quality benchmark, perplexity evaluation, or teacher-forced logit comparison was run.
The selector does not change score precision. The FP4 scoring accuracy tradeoff belongs to the
separate FP4 implementation, which is enabled on both sides here.

## Hardware and method

Two NVIDIA DGX Sparks, one GB10 per rank, tensor parallelism 2, CUDA 13.3.73,
driver 580.173.02, Zig 0.17.0. Both devices retained a **2200 MHz graphics/SM clock ceiling**.
The full run has 410 samples of both devices at approximately five-second intervals:
maximum observed frequency was 2190 MHz on each, with zero sample errors or observed violations.
This is a ceiling, not a fixed frequency, and sampling does not prove every intervening instant.

Model: `DeepSeek-V4.1-Flash-1017`, checkpoint
`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`, revision
`45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`.
Implementation commit: `a97b69afc9ea44e4655a7025a009e2ee26479249`.
Baseline branch commit: `d4b3e7f9197f83a3399df072120d7842ce50eac2`.

Temperature 0, seed 1234, thinking disabled, EOS ignored for a fixed output length. Each drafted
context request verifies zero cached prompt tokens and a cold cache state. Plain decode is an
immediate repeat with `draft=false`; its cache state is retained in the raw receipt. The near-1M
prompt leaves 256 positions for output and verification within the 1,048,576-position pool.

The candidate allocates **32 MiB of scratch per rank** before graph capture. The reported
rank-0 active allocator memory increased by exactly 33,554,432 bytes. Existing cache, model,
transport, and FP4 settings were retained. The candidate Docker image only replaces the native
server/worker binaries and sets `TF_DS_TOPK_EXACT=1` on top of the qualified native-FP4 image.

## Kernel results and implementation

Representative signed-int64 top-512 selections, one random row, CUDA graph replay:

| Entries | Original, microseconds | Candidate, microseconds | Speedup |
| --- | ---: | ---: | ---: |
| 16,384 | 44.39 | 19.03 | 2.33x |
| 32,768 | 78.26 | 24.16 | 3.24x |
| 65,536 | 144.77 | 25.92 | 5.59x |
| 131,072 | 271.57 | 28.58 | 9.50x |

These are selector timings, not model speedups. TensorFold already uses fused short-row
selection and exact tile-max pruning; long model contexts do not imply selection over the
entire context in this replacement. No full-model GPU phase trace was collected, so this
receipt does not establish top-k's share of total runtime. Some tiny boundary shapes showed
small microbenchmark regressions; all timings are retained in `topk-full.json`.

The CUDA implementation retains full 64-bit keys, skips common leading radix bytes, aggregates
histogram updates within warps, and merges local winners exactly. It preserves lower-index
tie ordering, negative-zero handling, visibility masking, duplicate integer values, and sorted
indices. The existing fused path and tile pruning remain. Unsupported shapes fall back to the
original selector. See [implementation and device checks](../../../tools/dsv41-zig/topk_exact/README.md).

## Evidence and reproduction

- `results.json`: full serial/concurrent token IDs, cache evidence, phase timings, retrieval checks, and descriptive summaries.
- `fixtures.json` and `fixtures/*.json.gz`: exact synthetic input arrays, positions, expected records, and hashes.
- `lifecycle.json`: cancellation and prefix-reuse checks.
- `clocks.jsonl.gz`: timestamped device samples.
- `source-proof.json` and `compiled-source-sha256.json`: compiled production sources match the Git commit after CRLF-to-LF normalization.
- `activation-proof.json`: both running candidate containers had native FP4 and exact top-k enabled.
- `baseline-fatbins.json`: hashes of the previously qualified fatbins before replacing `dsv41_ops.fatbin`.
- `build.py`: exact build recipe; the unchanged fatbins and FP4 image assets were reused to isolate this change.
- `orchestrator.py` and `lifecycle-helper.py`: executed deployment-specific A/B/A harness and lifecycle gate.
- `benchmark-helpers.json`: pinned source URLs and SHA256 hashes for the harness dependencies.
- `primary-restoration.json` and `worker-restoration.json`: original-image and health restoration checks.
- `host-tests.log`, `server-build.log`, `cpu-reference.log`, `memcheck.log`, `racecheck.log`, and the two lean logs: validation output.

The orchestration snapshot assumes the existing two-Spark deployment scripts and images; it
is not a generic installation script. Kernel tests are runnable using the commands in the
implementation README. Set `TF_DS_TOPK_EXACT=1` on both ranks to opt in; unset it to retain
the original selector. FP4 remains independently controlled by `TF_DS_INDEX_FP4_AOT`.
