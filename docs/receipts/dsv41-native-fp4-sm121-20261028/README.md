# Native FP4 indexer: two DGX Sparks, 2200 MHz ceiling

The native FP4 path reduced near-1M cold first-token latency by 6.24% in this A/B/A run. Short-context latency was effectively unchanged and decode regressed in some cases. The feature stays opt-in; the original service was restored after testing.

Each entry below is baseline ¡ú native FP4. Baseline is the mean of the before/after phase medians. The 2K and 64K phases each have three fixtures; near 1M has one fixture per phase. Every context reply is 64 generated tokens, greedy, with thinking disabled and `ignore_eos=true`.

| Prompt tokens | Cold TTFT (s) | Drafted decode (tok/s) | Plain decode (tok/s) | Lookup, each phase |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1.123 ¡ú 1.130 | 77.775 ¡ú 76.445 | 44.750 ¡ú 44.175 | 9/9 |
| 65,536 | 28.049 ¡ú 28.160 | 65.564 ¡ú 64.591 | 43.954 ¡ú 41.435 | 9/9 |
| 1,048,320 | 539.888 ¡ú 506.205 | 69.341 ¡ú 71.195 | 40.521 ¡ú 40.653 | 3/3 |

The near-1M prompt contains 1,048,320 tokens inside a 1,048,576-position pool, leaving output and verification reserve. Plain decode follows an immediate repeat and can reuse a retained prefix; its TTFT is not included in cold-prefill comparisons. The cold requests were verified to have zero cached prompt tokens.

At near 1M, baseline TTFT was 542.245 s before and 537.531 s after, versus 506.205 s for FP4. This is 6.24% lower latency against the bracketed baseline, or 6.65% higher input tokens per second. Drafted decode improved 2.67%; plain decode improved only 0.33%, below baseline decode drift. At 64K, plain decode fell 5.73%. These are descriptive measurements, not confidence intervals.

All 18 candidate replies matched the first baseline token for token; the restored baseline also matched (36 cross-phase comparisons). Drafted/plain and concurrent/solo checks passed. The lookup task has three eight-digit records at approximately 10%, 50%, and 90% of each prompt. All phases retrieved every record. This measures synthetic lookup and exact replay, not broad model quality or perplexity.

The lifecycle check cancelled one request during prefill and another after decoding began, then compared subsequent full replies and retained-prefix replay with the baseline. All five replies matched full token IDs.

Both devices were capped at 2200 MHz. 424 complete two-device clock samples recorded maxima of 2190 MHz and 2190 MHz, with zero cap violations or sample errors. The ceiling was reaffirmed after the original service was restored.

## Artifacts

- `results.json`: settings, software/checkpoint revisions, phase samples, full returned token IDs, accuracy, and provenance.
- `fixtures.json` and `fixtures/*.json.gz`: exact public token arrays, checksums, lookup values, and record positions.
- `clocks.jsonl.gz`: anonymized timestamped samples from both devices, nominally every five seconds.
- `lifecycle.json`: cancellation/replay checks with full observed IDs and baseline equality.
- `kernel-gates.json`: 113 passing score cases, query-bit checks, FP64 reference checks, and shape/variant coverage.
- `host-tests.log`, `memcheck.log`, `racecheck.log`: 218/218 host tests, zero memory errors, zero race hazards.
- `rejected-pre-fma-fix.json`: retained failed arithmetic receipt from before restoring the exact FMA operand order.
- `orchestrator.py`: the exact preflight/A/B/A controller used, including automatic restoration.
- `benchmark-helpers.json`: source URLs and SHA256 values for its standard-library benchmark dependencies.

## Reproduce the served run

Build the opt-in catalog and both binaries as described in `tools/dsv41-zig/fp4_index/README.md`. Stage an image derived from the same production image on both ranks, including the compiled catalog, and set `TF_DS_INDEX_FP4_AOT` inside that image. Keep the checkpoint, TP2 recipe, and 2200 MHz cap identical.

Place the helper files identified in `benchmark-helpers.json` under `bench-tools/` next to `orchestrator.py`, verifying their SHA256 values. The controller expects the matching two-Spark recipe under the home directory and its existing user service. It performs preflight unless `--apply` is explicitly supplied.

```sh
python3 orchestrator.py --candidate-image YOUR_IMAGE --contexts 2048,65536,1048320 --reps 3
python3 orchestrator.py --apply --candidate-image YOUR_IMAGE --contexts 2048,65536,1048320 --reps 3 --max-runtime-minutes 60
```

A new run generates fresh public fixtures; the archived token arrays above reproduce the measured inputs. The original EXL3 weights, packed key-cache layout, and allocated query stride are unchanged. This branch qualifies CUDA SM121 only. The lean checker remains at the inherited 1490 findings with zero new findings.
