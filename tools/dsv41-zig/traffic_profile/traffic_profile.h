#pragma once
#include <stdint.h>

// Optional host instrumentation; never put these calls inside graph capture.
// Load this library only for measurements. Each process profiles its current CUDA
// context; the caller owns cross-rank coordination and must preserve graph/PDL work.
// TF_TRAFFIC_ENABLE=1, TF_TRAFFIC_OUTPUT=/path/result-%r.jsonl, and
// TF_TRAFFIC_METRICS=comma,separated,metric.names are required. %r becomes rank;
// without %r, .rank<N>.jsonl is appended. Output is append-only JSONL.
// TF_TRAFFIC_RANGES=8 and TF_TRAFFIC_SKIP=2 apply independently to context buckets
// <=4096, <=131072, and larger. TF_TRAFFIC_MIN_CONTEXT/MAX_CONTEXT filter tokens;
// TF_TRAFFIC_MAX_ROWS=16 excludes accidental prefill calls. TF_TRAFFIC_BUCKET and
// TF_TRAFFIC_RUN are optional metadata. Call finish before prefill or teardown
// while the CUDA context is still current, especially when fewer than RANGES ran.
// begin/end: 0 skipped, 1 collected, -1 failed (inference may continue).
// No replay is performed; configurations needing more than one pass are rejected.
#ifdef __cplusplus
extern "C" {
#endif
int tf_traffic_begin(uint64_t context_tokens, uint32_t rows, int rank);
int tf_traffic_end(void);
// Invalidate the current window after failed verification; disables further collection without replay.
void tf_traffic_abort(void);
void tf_traffic_finish(void);
// Requires a current CUDA context. Enumerates metadata, without enabling counters
// or launching kernels. Comma-separated substrings filter base metrics.
int tf_traffic_query(const char *filters);
// Host-only enumeration: no CUDA initialization/context, no availability filtering.
int tf_traffic_query_chip(const char *chip, const char *filters);
#ifdef __cplusplus
}
#endif
