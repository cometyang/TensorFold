# Context-calibrated speculative decode measurements

These standard-library tools measure one already running TensorFold configuration.
They do not start services, change GPU clocks, or alter the engine's precision.
Use the same model, native FP4 kernels, two-node placement and 2200 MHz cap for
every baseline, calibration, candidate and plain-output run. Record clock/power
telemetry separately on both hosts throughout each measurement. Keep competing
requests out of the service; counter deltas cannot attribute unrelated traffic.

## Native runtime settings

All three controls are opt-in environment variables on the rank-0 native server:

- `TF_DS_SPEC_TRACE=/path/to/trace.jsonl` records buffered CPU wall timings,
  context lengths, confidence logits and potential accepted prefixes. It adds no
  GPU fence and records numeric trace IDs rather than request text or token IDs.
- `TF_DS_SPEC_SWEEP=1` requires tracing and cycles K=0 through K=5 in blocks of
  16 verification rounds. Use it only to collect calibration data; turn it off
  for performance comparisons and normal service.
- `TF_DS_SPEC_COSTS=/path/to/context-costs.json` loads the fitter's schema-version-1
  `dsv41` profiles. The core owns a copy of their costs. Invalid, empty or
  overlapping catalogs fail startup. Gaps, unmeasured concurrent stream counts,
  boundary-straddling proposals and insufficient row coverage retain the existing
  allocation policy. Without this variable the allocator is unchanged.

The loader validates schema, family, ranges and finite positive row times.
Deployment metadata in `provenance` is documentary; it is not matched against the
running hardware automatically. Select a catalog measured with the same model,
checkpoint, tensor-parallel placement, kernels and clock settings. Tracing and
the sweep are disabled for final throughput measurements.

## Frozen fixtures

Build the fixtures once, then reuse their exact token arrays in every configuration:

```sh
python3 tools/dsv41-zig/spec_calibration/fixtures.py http://127.0.0.1:8080 local-model \
  --split train --seed 2001 --output measurements/train
python3 tools/dsv41-zig/spec_calibration/fixtures.py http://127.0.0.1:8080 local-model \
  --split holdout --seed 7001 --output measurements/holdout
```

Defaults are 2,048, 65,536 and 1,047,040 input tokens, with prose, code and retrieval
fixtures at each length. They use the server's chat framing with thinking disabled,
deterministic synthetic reference material and a final instruction. Prose and code
use different training and holdout tasks. Three retrieval records appear near 10%,
50% and 90% of the filler. The builder records their actual token offsets and
expected codes. Filler consists of a shuffled bank of 64 original paragraphs or
code snippets; this controlled workload is not a broad real-world quality suite.

The near-window input leaves 1,536 slots: 1,024 generated tokens and 18 verifier
rows fit safely. Existing 1,048,320-token/128-output fixtures can be retained as a
separate continuity experiment; their result must not be pooled with the longer
protocol. Each fixture stores the full input IDs and SHA256 over little-endian u32
IDs. The same digest convention is used for full output IDs.
The native server's separate 12-character `token_sha` hashes comma-separated
decimal IDs; the client validates that convention separately and retains it as
`native_token_sha`. These two hash formats must not be compared as prefixes.

## Calibration and fitting

Collect training traces with the instrumented engine and its development-only
fixed-K sweep. The server's fixed-K environment setting is global, not an HTTP
request parameter. Retain configuration metadata, trace files and full outputs.
Use ordinary dynamic speculation for measured holdout performance afterward.

```sh
python3 tools/dsv41-zig/spec_calibration/fit.py measurements/train-trace.jsonl \
  --output measurements/context-costs.json
```

The fitter consumes schema-version-1 verify/draft/release JSONL. It groups by
maximum proposed context end, active stream count and actual verifier rows. It
also requires the maximum pending-only end to lie in the same range, so a wider
proposal cannot move otherwise uncovered measurements into a calibrated range. It
excludes plain `draft:false` rounds, each request's first three rounds, bucket
transitions and the first two remaining occurrences of each source/bucket/shape.
`forward_ms` is the median of `verify_ns / 1e6`: measured forward, absorb and
existing waits, without adding GPU synchronization.

Explicit `overhead_ms` is the median gap from one verify start to the next minus
the earlier verify duration. It includes drafting, communication, readback,
sampling and scheduling. The fitter requires adjacent retained rounds with the
same request/slot set, source and context range, and resets across releases,
warmups, changed draft request sets and range changes. It never reuses an overhead
learned by subtracting the old, context-independent table. It reports medians,
median absolute deviations, counts, exclusions and input hashes for audit.

Each emitted profile requires five measurements per needed row and five overhead
samples. By default a one-stream profile needs rows 1–6, and a two-stream profile
needs rows 2–12. Missing widths are not interpolated. Two-stream global fixed-K
sweeps often miss odd widths; collect mixed natural widths or leave that profile
absent. The unused row-1 entry for two streams repeats row 2 solely to make the
JSON array dense. The default inclusive ranges are 2,048–4,096, 65,536–131,072 and
1,047,040–1,048,576; unrepresented ranges and stream counts use the engine fallback.
These ranges are a modeling assumption; the receipt records the contexts actually
measured. Pass `--ranges` to narrow them. Do not fit on holdout traces.

An empty profile file is accompanied by exit status 2. An incomplete profile's
missing rows and overhead coverage remain in `provenance`. A partial set of complete
profiles can be used because missing contexts fall back. Each trace copied from a
live process should contain the release records for the measured requests; a final
summary appears only at process shutdown. A missing or truncated trace is not
evidence of an end-to-end timing win.

Inspect component timing, confidence calibration and the choices implied by the
two cost tables with the optional replay tool:

```sh
python3 tools/dsv41-zig/spec_calibration/analyze.py measurements/train-trace.jsonl \
  --catalog measurements/context-costs.json --legacy-overhead-ms 0 \
  --output measurements/training-analysis.json
```

Replay evaluates both policies under the measured costs and labels its gains as
counterfactual estimates. It cannot predict tokens on an unverified draft tail or
replace an end-to-end benchmark. The legacy overhead is a scenario input, not a
measurement of the running baseline; compare plausible values explicitly.

## Speed and exactness

Run an identical command for baseline and candidate, changing only the externally
selected server configuration, output path and label:

```sh
python3 tools/dsv41-zig/spec_calibration/benchmark.py http://127.0.0.1:8080 local-model \
  measurements/holdout/holdout-prose-2048-7001.json --label baseline \
  --tokens 1024 --temperature 0 --seed 19 --reps 3 --warmup-reps 1 \
  --output measurements/baseline.jsonl
```

Repeat for each context and prose/code/retrieval fixture, at temperatures 0 and
0.7, and with multiple fixed seeds. `--plain` gives the same engine's
`draft:false` exactness reference. Natural EOS is the default. `--ignore-eos`
creates a separate fixed-length speed protocol; use the identical setting in
all compared requests and report it. Passing two fixture paths launches two
streams together. Test solo and concurrent exactness separately. Optional API
credentials come from `TENSORFOLD_BENCH_API_KEY` and are not written to results.

Every JSONL record is saved immediately after one batch. It retains usage, full
output IDs, hashes, text, reasoning, sampling settings, chunk arrival times,
runtime timings, speculative counts and before/after server metrics. Client
decode rate is `(committed tokens - 1)/(last text chunk - first text chunk)`;
the first generated token is excluded. Hidden special tokens or text buffering
can distort this convention. Use the recorded runtime `tokens_per_second` and
`server_decode_tps_per_stream` as corroboration. The latter uses the server's
decode-seconds counter, averaged across active streams, and includes reply
cleanup; it is not aggregate throughput for concurrency. Client aggregate
throughput spans the first to last visible chunk across the batch. A single
text chunk is unmeasured, not a speed result.

Counter deltas report drafted tokens verified, accepted tokens actually kept,
and decode rounds for finished requests. Trace `potential_accepted_drafts` is
before EOS/budget clipping and must not replace those counters. Missing/reset
counters or unexplained generated tokens mark the deltas invalid.

```sh
python3 tools/dsv41-zig/spec_calibration/compare.py measurements/baseline.jsonl \
  measurements/candidate.jsonl --output measurements/comparison.json
```

The comparison excludes explicitly marked warmup batches from speed summaries,
uses matching input hashes/settings and reports medians and every sample. A
speed cell keeps plain and speculative requests separate; token exactness can
still compare those modes. Rates require valid counter attribution, completed
requests and positive finite throughput; excluded samples are counted by reason. A
20% throughput gain means candidate tok/s ≥ 1.20 × baseline tok/s; equivalently,
decode time for the same work falls by at least 16.67%. Three repetitions are a
minimum diagnostic, not a statistical confidence guarantee. Alternate baseline
and candidate configurations where practical and preserve a final baseline
recheck for drift. The speed target must be met on holdout requests, not only
on calibration traces or short graph-capture runs.

Exactness compares full output ID arrays, including sampled requests with the
same seed/history. Missing IDs or incomplete streams are unverified. Retrieval
checks report whether each expected code occurs in the response. This checks
scheduler output preservation and a synthetic task; it does not measure general
model accuracy or substitute for teacher-forced numerical diagnostics.

```sh
python3 -m unittest discover -s tools/dsv41-zig/spec_calibration -v
```
