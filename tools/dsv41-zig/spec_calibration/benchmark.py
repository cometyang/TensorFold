"""Exact-token HTTP benchmark for one externally managed server configuration."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import urllib.request

COUNTERS = ("generation_tokens_total", "prompt_tokens_total", "mtp_drafted_total",
            "mtp_accepted_total", "decode_rounds_total", "request_decode_seconds_sum")


def digest_ids(ids):
    if not isinstance(ids, list) or any(type(n) is not int or not 0 <= n < 2**32 for n in ids):
        raise ValueError("token_ids must be a list of unsigned 32-bit integers")
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def request(base, path, body=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TENSORFOLD_BENCH_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TENSORFOLD_BENCH_API_KEY"]
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    return urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + path, data, headers), timeout=timeout)


def api(base, path, body=None):
    with request(base, path, body) as response:
        return json.load(response)


def metrics(base):
    with request(base, "/metrics", timeout=30) as response:
        raw = response.read().decode()
    out = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].startswith("tensorfold:") and "{" not in fields[0]:
            value = float(fields[1])
            if math.isfinite(value):
                out[fields[0].split(":", 1)[1]] = value
    return out


def idle_metrics(base, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = metrics(base)
        if out.get("requests_running") == 0 and out.get("requests_waiting") == 0:
            return out
        time.sleep(0.2)
    raise TimeoutError("server did not become idle")


def metric_delta(before, after, generated):
    if any(name not in before or name not in after for name in COUNTERS):
        return {"valid": False, "error": "missing_counters"}
    delta = {name: after[name] - before[name] for name in COUNTERS}
    if any(value < 0 for value in delta.values()):
        return {"valid": False, "error": "counter_reset", "delta": delta}
    if delta["generation_tokens_total"] != generated:
        return {"valid": False, "error": "unattributed_generation", "delta": delta}
    drafted, accepted = delta["mtp_drafted_total"], delta["mtp_accepted_total"]
    rounds = delta["decode_rounds_total"]
    return {"valid": True, "delta": delta,
            "acceptance": accepted / drafted if drafted else None,
            "generated_tokens_per_round": generated / rounds if rounds else None}


def stream(base, model, fixture, tokens=1024, temperature=0.0, seed=1234, draft=True,
           ignore_eos=False, gate=None):
    prompt = fixture["prompt"]
    if digest_ids(prompt) != fixture["prompt_sha256"]:
        raise ValueError("fixture token digest mismatch")
    if len(prompt) + tokens + 18 > fixture.get("context_window", 1048576):
        raise ValueError("prompt plus output and 18 verifier rows exceeds context window")
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": temperature,
            "seed": seed, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "draft": draft, "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"enable_thinking": False}}
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    out = {"fixture": fixture["name"], "split": fixture["split"], "kind": fixture["kind"],
           "prompt_sha256": fixture["prompt_sha256"], "context": len(prompt), "seed": seed,
           "temperature": temperature, "max_tokens": tokens, "draft": draft, "ignore_eos": ignore_eos,
           "complete": False, "decode_tps": None, "token_ids": None}
    if gate is not None:
        gate.wait(timeout=30)
    sent = time.perf_counter()
    pieces, content, reasoning, usage, runtime, speculative = [], [], [], {}, {}, {}
    try:
        with request(base, "/v1/completions", body) as response:
            for raw in response:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    out["complete"] = True
                    break
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    raise ValueError("server_stream_error")
                usage = chunk.get("usage") or usage
                runtime = chunk.get("tensorfold") or runtime
                speculative = chunk.get("speculative") or speculative
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = choice.get("text") or delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if not isinstance(text, str) or not isinstance(thought, str):
                        raise ValueError("invalid_text_chunk")
                    content.append(text)
                    reasoning.append(thought)
                    if text or thought:
                        pieces.append((time.perf_counter() - sent, len(text) + len(thought)))
                    if choice.get("finish_reason"):
                        out["finish_reason"] = choice["finish_reason"]
        n = usage.get("completion_tokens")
        ids = runtime.get("token_ids")
        if not out["complete"]:
            raise ValueError("incomplete_stream")
        if type(n) is not int or n < 0 or not isinstance(ids, list) or len(ids) != n:
            raise ValueError("missing_or_inconsistent_token_ids")
        sha = digest_ids(ids)
        claimed = runtime.get("token_sha")
        native_sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
        if claimed and (not isinstance(claimed, str) or len(claimed) != 12 or not native_sha.startswith(claimed)):
            raise ValueError("server_token_digest_mismatch")
        if usage.get("prompt_tokens") != len(prompt):
            raise ValueError("server_prompt_length_mismatch")
        out.update(tokens=n, token_ids=ids, token_sha256=sha, native_token_sha=claimed)
        if len(pieces) >= 4 and n > 1 and pieces[-1][0] > pieces[0][0]:
            out["decode_tps"] = (n - 1) / (pieces[-1][0] - pieces[0][0])
    except Exception as exc:
        out["error"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    text, thought = "".join(content), "".join(reasoning)
    out.update(usage=usage, runtime={k: v for k, v in runtime.items() if k != "token_ids"},
               speculative=speculative, pieces=pieces, elapsed_s=time.perf_counter() - sent, text=text, reasoning=thought,
               ttft_s=pieces[0][0] if pieces else None, sent_monotonic=sent,
               output_sha256=hashlib.sha256(hashlib.sha256(text.encode()).digest() +
                                            hashlib.sha256(thought.encode()).digest()).hexdigest())
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    out["cached_tokens"] = cached
    out["cache_state"] = ("cold" if cached == 0 else "reused") if type(cached) is int else "unverified"
    expected = fixture.get("expected", [])
    out["retrieval"] = {value: value in text for value in expected}
    return out


def run_batch(base, model, fixtures, **settings):
    before = idle_metrics(base)
    gate = threading.Barrier(len(fixtures)) if len(fixtures) > 1 else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fixtures)) as pool:
        jobs = [pool.submit(stream, base, model, fixture, gate=gate, **settings) for fixture in fixtures]
        runs = [job.result() for job in jobs]
    generated = sum(run.get("tokens", 0) for run in runs)
    after = idle_metrics(base)
    # Finished request counters may be published just after the idle gauge changes.
    deadline = time.monotonic() + 5
    while after.get("generation_tokens_total", 0) - before.get("generation_tokens_total", 0) < generated:
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
        after = metrics(base)
    result = {"runs": runs, "metrics_before": before, "metrics_after": after,
              "metrics": metric_delta(before, after, generated), "aggregate_decode_tps": None}
    if result["metrics"]["valid"]:
        seconds = result["metrics"]["delta"]["request_decode_seconds_sum"]
        result["server_decode_tps_per_stream"] = (generated - len(runs)) / seconds if seconds > 0 else None
    if all(run.get("decode_tps") and not run.get("error") for run in runs):
        first = min(run["sent_monotonic"] + run["pieces"][0][0] for run in runs)
        last = max(run["sent_monotonic"] + run["pieces"][-1][0] for run in runs)
        result["aggregate_decode_tps"] = sum(run["tokens"] - 1 for run in runs) / (last - first)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup-reps", type=int, default=1)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    fixtures = [json.loads(path.read_text()) for path in args.fixtures]
    if not 1 <= len(fixtures) <= 2 or min(args.tokens, args.reps) < 1 or args.warmup_reps < 0:
        parser.error("use one or two fixtures and positive tokens/reps")
    for rep in range(args.warmup_reps + args.reps):
        record = run_batch(args.base, args.model, fixtures, tokens=args.tokens, temperature=args.temperature,
                           seed=args.seed, draft=not args.plain, ignore_eos=args.ignore_eos)
        record.update(schema_version=1, label=args.label, rep=rep, warmup=rep < args.warmup_reps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps({"label": args.label, "rep": rep, "warmup": record["warmup"],
                          "decode_tps": record["aggregate_decode_tps"],
                          "errors": [run.get("error") for run in record["runs"]]}), flush=True)


if __name__ == "__main__":
    main()
