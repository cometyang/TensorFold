"""Check exact token IDs, cancellation recovery and retained prefixes on an already-running candidate."""

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit

import benchmark

TOKENS = 128
SEED = 1234


def compact(run):
    fields = ("fixture", "prompt_sha256", "context", "seed", "temperature", "max_tokens", "draft",
              "ignore_eos", "complete", "error", "token_ids", "token_sha256", "tokens", "finish_reason",
              "cached_tokens", "cache_state", "usage", "speculative", "elapsed_s", "ttft_s", "decode_tps")
    return {key: run[key] for key in fields if key in run}


def valid(run):
    ids = run.get("token_ids")
    return (run.get("complete") is True and not run.get("error") and isinstance(ids, list)
            and len(ids) == TOKENS and run.get("tokens") == TOKENS)


def equivalent(reference, actual):
    fields = ("prompt_sha256", "seed", "temperature", "max_tokens", "ignore_eos")
    return (valid(reference) and valid(actual) and all(reference.get(k) == actual.get(k) for k in fields)
            and reference["token_ids"] == actual["token_ids"])


def altered(fixture, token, suffix):
    out = dict(fixture, prompt=list(fixture["prompt"]), name=fixture["name"] + suffix)
    out["prompt"][0] = token if token != out["prompt"][0] else token + 1
    out["prompt_sha256"] = benchmark.digest_ids(out["prompt"])
    return out


def cancel(base, model, fixture, decode):
    before = benchmark.idle_metrics(base, timeout=120)
    parts = urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
        raise ValueError("base must be an HTTP(S) origin or base path")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = cls(parts.hostname, parts.port, timeout=180)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TENSORFOLD_BENCH_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TENSORFOLD_BENCH_API_KEY"]
    body = {"model": model, "prompt": fixture["prompt"], "max_tokens": 1024, "temperature": 0,
            "seed": SEED, "stream": True, "return_token_ids": True, "draft": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    result = {"fixture": fixture["name"], "context": len(fixture["prompt"]),
              "prompt_sha256": fixture["prompt_sha256"], "phase": "decode" if decode else "prefill",
              "admission_observed": False, "phase_observed": False, "connection_closed": False}
    response = None
    started = time.monotonic()
    try:
        connection.request("POST", parts.path.rstrip("/") + "/v1/completions",
                           json.dumps(body, separators=(",", ":")), headers)
        if decode:
            response = connection.getresponse()
            result["http_status"] = response.status
            if response.status != 200:
                raise ValueError("cancellation request did not return HTTP 200")
            for line in response:
                if line.strip() == b"data: [DONE]":
                    break
                if not line.startswith(b"data:"):
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    raise ValueError("cancellation stream returned an error")
                if any(choice.get("text") or (choice.get("delta") or {}).get("content")
                       or (choice.get("delta") or {}).get("reasoning_content")
                       for choice in chunk.get("choices", [])):
                    result.update(admission_observed=True, phase_observed=True)
                    break
        else:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = benchmark.metrics(base)
                if state.get("requests_running", 0) > 0:
                    result["admission_observed"] = True
                    time.sleep(0.25)
                    state = benchmark.metrics(base)
                    result["phase_observed"] = (state.get("requests_running", 0) > 0
                                                and state.get("generation_tokens_running") == 0)
                    result["observed_metrics"] = {key: state.get(key) for key in
                                                  ("requests_running", "generation_tokens_running")}
                    break
                time.sleep(0.05)
    except Exception as exc:
        result["error"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    finally:
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if response is not None:
            response.close()
        connection.close()
        result["connection_closed"] = True
        result["closed_after_s"] = time.monotonic() - started
    try:
        after = benchmark.idle_metrics(base, timeout=120)
        deadline = time.monotonic() + 5
        while after.get("client_disconnections_total", 0) <= before.get("client_disconnections_total", 0):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
            after = benchmark.metrics(base)
        result["idle_after"] = True
        result["disconnects_delta"] = (after.get("client_disconnections_total", 0)
                                       - before.get("client_disconnections_total", 0))
    except Exception as exc:
        result.update(idle_after=False, error=type(exc).__name__)
    result["passed"] = (result["phase_observed"] and result.get("idle_after", False)
                        and result.get("disconnects_delta", 0) > 0 and not result.get("error"))
    return result


def check(base, model, fixtures, output_path):
    output_path = Path(output_path)
    shorts = [fixture for fixture in fixtures if len(fixture["prompt"]) == 2048]
    longs = [fixture for fixture in fixtures if len(fixture["prompt"]) == 65536]
    if not shorts or not longs:
        raise ValueError("lifecycle needs original 2048-token and 65536-token fixtures")
    for fixture in shorts + longs:
        if benchmark.digest_ids(fixture["prompt"]) != fixture["prompt_sha256"]:
            raise ValueError("fixture token digest mismatch")
    first = shorts[0]
    second = next((item for item in shorts if item["prompt_sha256"] != first["prompt_sha256"]), first)
    long = longs[0]
    result = {"schema_version": 1, "model": model, "tokens": TOKENS, "seed": SEED,
              "checks": [], "cancellations": [], "runs": [], "passed": False,
              "scope": "2K shared/solo greedy and sampled; 2K/64K cancellation and prefix lifecycle"}

    def save():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
        temporary.replace(output_path)

    def record(label, batch_fixtures, temperature=0.0, draft=True):
        batch = benchmark.run_batch(base, model, batch_fixtures, tokens=TOKENS, temperature=temperature,
                                    seed=SEED, draft=draft, ignore_eos=True)
        rows = batch["runs"]
        accepted = all(valid(run) for run in rows) and (batch.get("metrics") or {}).get("valid") is True
        result["runs"].append({"label": label, "runs": [compact(run) for run in rows],
                               "metrics": batch.get("metrics")})
        result["checks"].append({"label": label + ":complete-and-attributed", "passed": accepted})
        save()
        if not accepted:
            raise RuntimeError("incomplete or unattributed lifecycle request: " + label)
        return rows

    def compare(label, reference, actual):
        result["checks"].append({"label": label, "fixture": actual["fixture"],
                                  "same_full_token_ids": equivalent(reference, actual),
                                  "passed": equivalent(reference, actual)})
        save()

    save()
    try:
        short_reference = None
        for temperature in (0.0, 0.7):
            mode = "greedy" if temperature == 0 else "sampled"
            solo = [record(mode + ":solo:" + str(i), [fixture], temperature)[0]
                    for i, fixture in enumerate((first, second))]
            paired = record(mode + ":concurrent", [first, second], temperature)
            for i in range(2):
                compare(mode + ":concurrent-equals-solo:" + str(i), solo[i], paired[i])
            if temperature > 0:
                for i, fixture in enumerate((first, second)):
                    plain = record(mode + ":plain:" + str(i), [fixture], temperature, draft=False)[0]
                    compare(mode + ":draft-equals-plain:" + str(i), plain, solo[i])
            if temperature == 0:
                short_reference = solo[0]
        long_reference = record("64K:original-before-cancellation", [long])[0]
        long_plain = record("64K:sampled-plain", [long], temperature=0.7, draft=False)[0]
        long_draft = record("64K:sampled-drafted", [long], temperature=0.7)[0]
        compare("64K:sampled-draft-equals-plain", long_plain, long_draft)
        cancelled = altered(long, 999, "-cancel-prefill")
        result["cancellations"].append(cancel(base, model, cancelled, False))
        save()
        redo = record("2K:after-prefill-cancellation", [first])[0]
        compare("2K:prefill-cancellation-recovery", short_reference, redo)
        result["cancellations"].append(cancel(base, model, first, True))
        save()
        redo = record("2K:after-decode-cancellation", [first])[0]
        compare("2K:decode-cancellation-recovery", short_reference, redo)
        redo = record("64K:original-after-cancellation", [long])[0]
        compare("64K:original-retained-agreement", long_reference, redo)
        for fixture, token in ((first, 1001), (long, 1002)):
            fresh = altered(fixture, token, "-fresh-lifecycle")
            cold = record(str(len(fresh["prompt"])) + ":fresh-prefix", [fresh])[0]
            applicable = len(fresh["prompt"]) >= 4096
            retained = record(str(len(fresh["prompt"])) + ":prefix-replay", [fresh])[0]
            compare("fresh-replay-token-agreement", cold, retained)
            result["checks"].append({"label": "fresh-retained-cache-observed" if applicable else "short-prefix-replay",
                                      "fixture": fresh["name"], "retention_applicable": applicable,
                                      "reason": None if applicable else "default min_prompt=4096 keeps no 2K snapshot",
                                      "first_token": fresh["prompt"][0], "fresh_cached_tokens": cold.get("cached_tokens"),
                                      "retained_cached_tokens": retained.get("cached_tokens"),
                                      "passed": cold.get("cached_tokens") == 0
                                      and type(retained.get("cached_tokens")) is int
                                      and (retained["cached_tokens"] > 0 if applicable else retained["cached_tokens"] == 0)})
            save()
        result["passed"] = (all(item["passed"] for item in result["checks"])
                            and all(item["passed"] for item in result["cancellations"]))
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.base, args.model, [json.loads(path.read_text()) for path in args.fixtures], args.output)
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]),
                      "cancellations": result["cancellations"], "error": result.get("error")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
