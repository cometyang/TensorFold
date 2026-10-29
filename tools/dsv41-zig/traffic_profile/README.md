# Verification-round memory counters

This optional CUPTI library collects aggregate hardware counters around real
verification rounds, including their CUDA graphs. It requires a single-pass
metric configuration and never replays model or communication work. It preserves
kernel concurrency inside each range. This is measurement instrumentation;
measure tokens/second separately with the library disabled.

## Prefetch experiments

The following optional controls are parsed once before CUDA graph capture.
They preserve the existing defaults and do not change model arithmetic.
They take effect only when decode L2 prefetch is enabled.

| Setting | Default | Meaning |
| --- | --- | --- |
| `TF_DS_L2_PREFETCH_MASK` | `31` | Stage bits: attention output 1, MoE 2, next layer 4, Engram 8, head 16 |
| `TF_DS_L2_PREFETCH_BUDGET_PERCENT` | `100` | Percentage of each existing byte budget, rounded down to 128-byte lines |
| `TF_DS_L2_PREFETCH_RATE_GBPS` | `230` | Requested issue pacing, integer 1–1000 GB/s; not measured memory bandwidth |

For example, mask `30` omits attention-output prefetch, and mask `0` omits all
prefetch launches. A rate of `115` retains the byte budgets but doubles the
interval between 1 MiB waves. The existing 3 microsecond startup delay stays
unchanged. Invalid settings fail before capture. Compare throughput with
profiling disabled, and retain a return-to-baseline control before selecting a
setting; fewer prefetch bytes do not necessarily make inference faster.

## Build on DGX Spark

The tested toolchain is CUDA 13.3, GNU C++17, and the CUDA SBSA libraries. No nvcc
compilation is required; the smoke test loads a small PTX kernel through the
driver API.

```bash
bash tools/dsv41-zig/traffic_profile/build.sh /tmp/tf-traffic
```

`CUDA_HOME`, `CUDA_TARGET_DIR`, and `CXX` override the defaults. The following
explicit commands were used for the validated worker build:

```bash
g++ -std=c++17 -O2 -fPIC -shared traffic_profile.cpp \
  -I/usr/local/cuda-13.3/targets/sbsa-linux/include \
  -L/usr/local/cuda-13.3/targets/sbsa-linux/lib \
  -L/usr/local/cuda-13.3/targets/sbsa-linux/lib/stubs \
  -Wl,-rpath,/usr/local/cuda-13.3/targets/sbsa-linux/lib \
  -lcuda -lcupti -pthread -o libtf_traffic.so

g++ -std=c++17 -O2 traffic_probe.cpp traffic_profile.cpp \
  -I/usr/local/cuda-13.3/targets/sbsa-linux/include \
  -L/usr/local/cuda-13.3/targets/sbsa-linux/lib \
  -L/usr/local/cuda-13.3/targets/sbsa-linux/lib/stubs \
  -Wl,-rpath,/usr/local/cuda-13.3/targets/sbsa-linux/lib \
  -lcuda -lcupti -pthread -o traffic_probe
```

The stub directory is used only at link time. Runtime library search must resolve
the real NVIDIA driver. CUPTI also needs the compatible NVIDIA profiling
libraries supplied with that CUDA installation.

The build script honors an existing `CUDA_HOME`. If it names `/usr/local/cuda`,
that alias is embedded in RUNPATH; the explicit commands above embed the versioned
path. This can change the binary hash even when both paths select CUDA 13.3 and
the C++ sources are identical. Record the compiler arguments and artifact hashes.

## Discover counters before collecting

```bash
# Host-only query: no CUDA initialization, allocations, or workload.
./traffic_probe --chip GB20B dram,sysmem,c2c,lts__d_sectors

# Query the actual current device and counter availability. Creates a CUDA
# context, but allocates no test buffers and launches no kernels.
./traffic_probe --query dram,sysmem,c2c,lts__d_sectors
```

GB10 reports the CUPTI chip name `GB20B`. The installed CUDA 13.3 counter catalog
exposes the following useful bundle, verified to fit one hardware pass:

| Metric | What its value measures |
| --- | --- |
| `lts__d_sectors_fill_sysmem.sum` | L2 sectors filled from system memory |
| `lts__t_sectors_aperture_sysmem_op_read.sum` | L2 read-request sectors for the system-memory address aperture |
| `lts__t_sectors_aperture_sysmem_op_write.sum` | L2 write-request sectors for that aperture |
| `gpu__time_duration.sum` | GPU range duration, in nanoseconds |

Each L2 sector is 32 bytes. Multiply sector counts by 32 to obtain the bytes
represented by that counter. See NVIDIA's [metric quantity definitions and L2
memory tables](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference).

These are measured GPU memory-demand counters. The two request counters include
L2 hits; write requests do not measure bytes physically written back to LPDDR.
System-memory fills provide a closer indication of external read traffic, but
coherent CPU-cache responses and other traffic prevent equating them with total
LPDDR bus bytes. This GB20B catalog exposed no `dram__*`, C2C/NVLink byte, or
dirty-eviction/writeback counters in the queried families. Do not label the sum of
these three sector counters as total DRAM traffic: that would double-count reads.

CUPTI reports context scope for these metrics, but NVIDIA documents that other
contexts can affect L2/device-memory counters. Keep other GPU activity idle and
record that limitation. See [CUPTI profiling known
issues](https://docs.nvidia.com/cupti/release-notes/release-notes.html#profiling).

Setting `TF_TRAFFIC_METRICS` during either query also emits the configuration's
pass count without enabling collection. Host-only queries do not test runtime
counter permissions or device availability.

## Enable a bounded measurement

The library defaults to off. Use a temporary measurement launch on both ranks:

```bash
export TF_DS_TRAFFIC_LIBRARY=/path/libtf_traffic.so
export TF_TRAFFIC_ENABLE=1
export TF_TRAFFIC_OUTPUT=/path/traffic-%r.jsonl
export TF_TRAFFIC_METRICS='lts__d_sectors_fill_sysmem.sum,lts__t_sectors_aperture_sysmem_op_read.sum,lts__t_sectors_aperture_sysmem_op_write.sum,gpu__time_duration.sum'
export TF_TRAFFIC_SKIP=16
export TF_TRAFFIC_RANGES=16
export TF_TRAFFIC_RUN=baseline
```

`%r` expands to the process's rank. Without it, `.rank<N>.jsonl` is appended.
Files are append-only; use a fresh output path for each run.

The skip and range limits apply independently to three context buckets, for the
process lifetime:

| Bucket | Context tokens | Intended benchmark case |
| --- | --- | --- |
| `b0` | 0–4,096 | 2K |
| `b1` | 4,097–131,072 | 64K |
| `b2` | Above 131,072 | Near 1M, when that prompt is supplied |

The buckets do not manufacture or assert prompt lengths; each receipt includes
the actual `context_tokens`. Defaults are two skipped rounds and eight collected
rounds per bucket. `TF_TRAFFIC_MIN_CONTEXT` and `TF_TRAFFIC_MAX_CONTEXT` optionally
filter eligibility; `TF_TRAFFIC_MAX_ROWS` defaults to 16. `TF_TRAFFIC_BUCKET` is a
metadata label, not a selector. `TF_TRAFFIC_RUN` labels the configuration receipt.

The host calls `tf_traffic_begin(context_tokens, rows, rank)` before verification
and `tf_traffic_end()` after its existing completion boundary. Calls must be made
on the same host thread with the same current CUDA context, outside graph capture.
After the configured number of ranges the library stops, decodes, and disables
the session. TensorFold also calls `Collector.finish()` from `Model.fillBegin`
and `Model.fillChunk`, after input validation and before prefill work, and during
model teardown. Both ranks execute these existing fill commands, including when
a fill chunk follows interleaved decode. These hooks close a partial window before
the next prefill without adding a wire protocol message. They do not reset the
per-bucket counters. Request release itself does not flush the collector; a final
partial window is flushed by the next fill or model teardown while the CUDA
context remains alive.

On the tested deployment, querying hardware availability without `SYS_ADMIN`
returned `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. Add `--cap-add SYS_ADMIN` only to
the temporary profiling containers. The collector requires no privileged
container and makes no host driver-security changes. Compile in a CPU-only
container; a metadata/smoke container needed a 4 GiB memory limit for CUDA context
initialization on the tested system. Neither the library nor probe changes GPU
clocks: apply and monitor the agreed 2200 MHz caps separately.

## Validate receipts and accuracy

Before enabling counters, the library checks `numOfPasses == 1`. It selects
`CUPTI_UserRange` and `CUPTI_UserReplay`, with one nesting level. After collection
it requires an entirely submitted pass, zero dropped ranges, matching range
counts and identities, and finite metric values. It buffers results until the
window validates. An error disables measurement and returns `-1`; it never
replays work or aborts inference. Treat an `error` record or a missing expected
`window_complete` record as an invalid measurement. See the [CUPTI range
profiling API](https://docs.nvidia.com/cupti/main/main.html#cupti-range-profiling-api).

An isolated validation run, with the served API idle, can exercise a 64 MiB
read/modify/write buffer larger than GB10's L2 cache:

```bash
TF_TRAFFIC_ENABLE=1 TF_TRAFFIC_SKIP=1 TF_TRAFFIC_RANGES=4 \
TF_TRAFFIC_OUTPUT=/path/smoke-%r.jsonl \
TF_TRAFFIC_METRICS="$TF_TRAFFIC_METRICS" ./traffic_probe --smoke64MiB
```

It executes one warmup and six real graph launches, with no replay. The final
buffer value must be 7. The initial DGX Spark validation captured four valid
ranges: system-memory fills were 67.110–67.297 MB per range for a 67.109 MB buffer;
L2 read/write requests had comparable size. The small excess is a reminder that
counter scope and background activity matter. `--smoke` uses a 1 MiB buffer that
can remain in L2 and therefore is unsuitable for checking external-read volume.

Do not run Nsight Compute, Nsight Systems, or another CUPTI profiler alongside
this collector. Retain both-rank counter files, clock records, actual row counts,
and accepted-token counts. Use separate unprofiled runs for throughput and output
accuracy comparisons; profiling durations alone are not speed benchmarks.

## Analyze collected receipts

The standard-library Python analyzer reads a controller `receipt.json` with
`phases`, each containing request batches and relative `counter_files` entries.
Keep the counter files beside the receipt in their recorded subdirectories.
It makes no network requests and requires no CUDA installation.

Receipts and references may be `.json.gz`; counter and trace files may be
`.jsonl.gz`. If a receipt names an uncompressed relative file that is absent,
the analyzer looks for the same name with `.gz` appended. Both paths must remain
inside the receipt directory after resolving links. Digests are checked against
the decompressed original bytes, so compression does not change the analysis.

```bash
python3 tools/dsv41-zig/traffic_profile/analyze.py /path/run/receipt.json \
  --reference-receipt /path/reference/receipt.json \
  --join-spec-trace --output /path/run/analysis.json
python3 -m unittest discover -s tools/dsv41-zig/traffic_profile -p 'test_*.py'
```

`--reference-receipt` supplies frozen plain-output token arrays for exact prefix
comparisons. Without it, the analyzer uses plain outputs in the current receipt
and identifies self comparisons separately. Controller exactness claims are
reported separately from independently compared token arrays. Missing references
remain unknown.

The report groups counter samples by phase, requested context, rank and actual
verification row count. It reports sample counts, means and medians for system
memory fills, L2 read/write requests and GPU range duration. Inferred fill
bandwidth divides fills by range duration; it is not total physical DRAM
bandwidth. The counters overlap and are never added together.

Unprofiled speed comparisons exclude warmups and all profiling requests. The
baseline is the arithmetic mean of A1 and A2 per-fixture median throughput;
context aggregates use equal-weight geometric means across fixtures. Use
`--baseline-labels BEFORE AFTER` to change the bracket names and
`--target-tps 80` to set the throughput target. Plain controls remain diagnostic.
Missing bracket phases do not produce an inferred performance gain.
Available output lengths, requested lengths, temperature and seed must agree
within each fixture across measured phases and plain controls. Conflicting
fixtures are excluded from speed comparisons and listed in `settings_conflicts`;
missing settings are not invented.

Optional `--join-spec-trace` uses the primary rank's `spec-trace.jsonl`. Each
`rankR_bB_roundK` range must match the trace's zero-based eligible verification
index in the same context bucket, with identical row count and maximum input
position. Eligibility follows the recorded context and row filters. Failed
joins remain unmatched and are listed in `issues`.

Joined potential yield is one new target token plus the accepted draft prefix
per window, before stop tokens and output-budget truncation. The pending input
was already emitted and is not counted again. Fill bytes per potential emitted
token cover target forward plus DSpark tap absorption; they exclude full draft
passes, argmax/readback and prefill. This denominator is not an observed committed
output count. Joined CPU wall times remain profiling diagnostics, not GPU time
or unprofiled throughput. Inspect `issues`, unmatched counts and sample sizes
before drawing conclusions from the report.
