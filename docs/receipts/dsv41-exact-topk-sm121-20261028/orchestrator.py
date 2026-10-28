"""Preflight by default; --apply runs A/B/A and restores the original user service."""
import argparse
import hashlib
import json
import os
import re
import random
from pathlib import Path
import signal
import shutil
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
import uuid

BASE_IMAGE = "tensorfold-dsv41:codex-native-fp4-73ee55ad604e"
ORIGINAL_IMAGE = "tensorfold-dsv41:exl3-tp2-625d420e24a4-ca20261022"
CANDIDATE_IMAGE = None
MODEL = "DeepSeek-V4.1-Flash-1017"
URL = "http://127.0.0.1:8888"
SERVICE = "dsv41-tensorfold.service"
RECIPE = Path.home() / "DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks"
ROOT = Path(__file__).resolve().parent


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def command(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def ready(timeout=360):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            health = api("/health")
            if health.get("status") == "ok" and not health.get("warming"):
                return health
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("API did not become ready")


def idle(timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        live = api("/health").get("live", {})
        if not live.get("connections", 0) and not live.get("waiting", 0):
            return
        time.sleep(0.25)
    raise RuntimeError("API is busy; no service stop is authorized for active requests")


def encode(text):
    return api("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]


def make_fixture(count, nonce, rep):
    marker = "FP4CONTENT" + nonce
    rendered = api("/tokenize", {"model": MODEL,
        "messages": [{"role": "user", "content": marker}],
        "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
    middle = encode(marker)
    locations = [i for i in range(len(rendered) - len(middle) + 1)
                 if rendered[i:i + len(middle)] == middle]
    if len(locations) != 1:
        raise ValueError("could not isolate chat content placeholder")
    first = locations[0]
    prefix, suffix = rendered[:first], rendered[first + len(middle):]
    rng = random.Random(nonce + str(count) + str(rep))
    expected = {label: str(rng.randrange(10000000, 100000000)) for label in ("amber", "cobalt", "tulip")}
    lead = encode(f"Public fixture {nonce} {count} {rep}. Three exact lookup records are mixed with unrelated notes.\n")
    question = encode("\nRead the three lookup records above. Return their exact eight-digit values, "
        "one per line, in this format: amber=VALUE, cobalt=VALUE, tulip=VALUE. No explanation.\n")
    fill = encode("GPU arithmetic combines rows and columns. A cache stores reusable data.\n")
    records = [encode("\nLOOKUP RECORD " + key + " = " + value + ".\n") for key, value in expected.items()]
    budget = count - len(prefix) - len(suffix) - len(question)
    content = list(lead)
    positions = []
    for fraction, record in zip((0.1, 0.5, 0.9), records):
        needed = int(budget * fraction) - len(content)
        if needed < 0:
            raise ValueError("fixture too short")
        content += (fill * ((needed + len(fill) - 1) // len(fill)))[:needed]
        positions.append(len(prefix) + len(content))
        content += record
    needed = budget - len(content)
    content += (fill * ((needed + len(fill) - 1) // len(fill)))[:needed]
    ids = prefix + content + question + suffix
    if len(ids) != count:
        raise ValueError("fixture length mismatch")
    return ids, {"expected": expected, "needle_token_positions": positions}


def main():
    global CANDIDATE_IMAGE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--candidate-image", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--contexts", default="2048,65536")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-runtime-minutes", type=int, default=55)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    CANDIDATE_IMAGE = args.candidate_image
    contexts = [int(value) for value in args.contexts.split(",")]
    if args.reps < 1 or min(contexts) < 32 or max(contexts) + 128 + 18 > 1048576:
        p.error("invalid repetitions or prompt plus output and verification reserve exceeds the pool")
    sys.path.insert(0, str(ROOT / "bench-tools"))
    import bench_concurrent as bench
    original_authorize = bench.authorize
    def authorize_with_ids(request, api_key):
        body = json.loads(request.data)
        body["return_token_ids"] = True
        request.data = json.dumps(body).encode()
        original_authorize(request, api_key)
    bench.authorize = authorize_with_ids
    class TextEvidence(bench.Evidence):
        def __init__(self):
            super().__init__()
            self.parts = []
        def add(self, choice):
            super().add(choice)
            delta = choice.get("delta") or {}
            self.parts.append(choice.get("text") or delta.get("content") or "")
        def finish(self, usage, runtime, complete):
            observed = {**super().finish(usage, runtime, complete), "text": "".join(self.parts)}
            ids = runtime.get("token_ids")
            if complete and isinstance(ids, list) and len(ids) == observed["tokens"] and all(
                    type(value) is int and 0 <= value < 2**32 for value in ids):
                observed["token_ids"] = ids
                observed["token_ids_sha256"] = hashlib.sha256(struct.pack("<" + "I" * len(ids), *ids)).hexdigest()
            return observed
    bench.Evidence = TextEvidence
    for image in (ORIGINAL_IMAGE, BASE_IMAGE, CANDIDATE_IMAGE):
        command(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL)
    command(["systemctl", "--user", "is-active", "--quiet", SERVICE])
    idle()
    run_dir = args.resume.resolve() if args.resume else ROOT / ("ab-" + time.strftime("%Y%m%dT%H%M%S"))
    if not run_dir.is_relative_to(ROOT.resolve()):
        raise ValueError("resume must be inside the experiment directory")
    prior = None
    if args.resume:
        prior = json.loads((run_dir / "receipt.json").read_text())
        contexts = prior["contexts"]
        args.reps = prior["reps"]
        shutil.copy2(run_dir / "receipt.json", run_dir / ("receipt-before-resume-" + time.strftime("%H%M%S") + ".json"))
    else:
        run_dir.mkdir()
    nonce = uuid.uuid4().hex
    expectations = json.loads((run_dir / "fixture-expectations.json").read_text()) if prior else {}
    fixtures = []
    for count in contexts:
        repetitions = 1 if count > 65536 else args.reps
        for rep in range(repetitions):
            name = f"context-{count}-rep-{rep}"
            if prior:
                ids = json.loads((run_dir / (name + ".json")).read_text())
            else:
                ids, expectations[name] = make_fixture(count, nonce, rep)
                (run_dir / (name + ".json")).write_text(json.dumps(ids))
            if len(ids) != count or any(type(value) is not int for value in ids):
                raise ValueError("invalid saved fixture")
            fixtures.append({"name": name, "kind": "completion", "prompt": ids, "count": count,
                             **expectations[name]})
    (run_dir / "fixture-expectations.json").write_text(json.dumps(expectations, indent=2) + "\n")
    report = {
        "schema_version": 1, "applied": args.apply, "contexts": contexts,
        "reps": args.reps, "large_context_reps": 1, "output_tokens": 128,
        "verification_reserve_positions": 18,
        "images": {"baseline": BASE_IMAGE, "candidate": CANDIDATE_IMAGE},
        "native_fp4_indexer": True, "exact_topk_candidate": True, "clock_ceiling_mhz": 2200, "phases": {}, "restored": False,
        "retrieval_records_per_fixture": 3, "fixture_expectations": expectations,
        "quality_evidence": "Output SHA256 plus client SHA256 of validated full returned token IDs",
    }
    if prior:
        report = prior
        report.setdefault("resumptions", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                     "reason": "wait for completed-request cleanup before next request"})
    path = run_dir / "receipt.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if args.smoke:
        item = next(fixture for fixture in fixtures if fixture["count"] == 2048)
        result = bench.stream(URL, MODEL, item, 64, 0, 1234)
        print(json.dumps({"smoke": True, "expected": item["expected"], **result}), flush=True)
        return
    if not args.apply:
        print(json.dumps({"preflight": "passed", "fixtures": len(fixtures), "receipt": str(path)}), flush=True)
        return
    changed = False

    def interrupted(signum, frame):
        raise SystemExit(f"received signal {signum}; restoring original service")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
        signal.signal(signum, interrupted)
    signal.alarm(args.max_runtime_minutes * 60)

    def measure(label):
        phase = report["phases"].setdefault(label, {"health_before": api("/health"), "requests": []})
        rows = phase["requests"]
        warmup = {"name": "warmup", "kind": "completion", "prompt": encode("Warmup GPU kernels.\n") * 256}
        if not rows:
            warm = bench.stream(URL, MODEL, warmup, 32, 0, 1234)
            if not warm.get("complete"):
                raise RuntimeError("warmup failed")
        specs = [(fixture, 128, draft) for fixture in fixtures for draft in (True, False)]
        specs += [(item, 128, True) for item in bench.PROMPTS]
        specs += [(item, 128, False) for item in bench.PROMPTS]
        for item, length, draft in specs:
            if any(row["name"] == item["name"] and row["draft"] == draft and row["complete"] for row in rows):
                continue
            idle()
            result = bench.stream(URL, MODEL, item, length, 0, 1234, draft=draft)
            keys = ("complete", "tokens", "prompt_tokens", "cached_tokens", "cache_state", "ttft_s",
                    "decode_tps", "output_sha256", "token_sha", "token_sha256", "error", "text", "token_ids_sha256", "token_ids")
            row = {"name": item["name"], "draft": draft, **{key: result.get(key) for key in keys}}
            if "expected" in item:
                text = result.get("text", "")
                row["retrieval"] = {label: bool(re.search(r"\b" + label + r"\s*[:=]\s*" + value + r"\b", text, re.I))
                                    for label, value in item["expected"].items()}
            rows.append(row)
            save()
            print(json.dumps({"phase": label, **row}), flush=True)
            if not row["complete"] or row["tokens"] != length:
                raise RuntimeError("incomplete benchmark request")
            if "count" in item and (row["prompt_tokens"] != item["count"] or (draft and row["cache_state"] != "cold")):
                raise RuntimeError("exact cold-prefill evidence failed")
        concurrent = bench.together(URL, MODEL, [(item, 1234) for item in bench.PROMPTS], 128, 0)
        solos = {row["name"]: row for row in rows if row["draft"]}
        concurrent_rows = []
        for result in concurrent:
            row = {key: result.get(key) for key in (
                "prompt", "complete", "tokens", "decode_tps", "output_sha256", "token_sha", "token_ids_sha256", "token_ids", "error",
            )}
            solo = solos[row["prompt"]]
            row["same_solo_output"] = row["output_sha256"] == solo["output_sha256"]
            row["same_solo_full_ids"] = bool(row.get("token_ids_sha256")) and row.get("token_ids") == solo.get("token_ids")
            row["same_solo_short_hash"] = bool(row["token_sha"]) and row["token_sha"] == solo["token_sha"]
            concurrent_rows.append(row)
        report["phases"][label]["concurrent"] = concurrent_rows
        report["phases"][label]["health_after"] = api("/health")
        save()

    def restore():
        nonlocal changed
        command(["bash", "stop.sh"], cwd=RECIPE, timeout=180)
        command(["systemctl", "--user", "start", SERVICE], timeout=60)
        ready()
        changed = False
        report["restored"] = True
        save()
        print("ORIGINAL_SERVICE_RESTORED", flush=True)

    def launch(image, label):
        command(["bash", "stop.sh"], cwd=RECIPE, timeout=180)
        environment = os.environ.copy()
        environment.update(IMAGE=image, PREPARE="0", FOREGROUND="0", WAIT_TIMEOUT="360",
                           TF_DS_PREFILL_WEIGHT_CACHE_MIB="0")
        with (run_dir / (label + "-launch.log")).open("w") as log:
            command(["bash", "start-tensorfold-site.sh"], cwd=RECIPE, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=480)
        ready()

    try:
        idle()
        changed = True
        command(["systemctl", "--user", "stop", SERVICE], timeout=240)
        launch(BASE_IMAGE, "baseline-before")
        measure("A1")
        idle()
        launch(CANDIDATE_IMAGE, "candidate")
        measure("B")
        import lifecycle
        report["lifecycle"] = lifecycle.check(sys.modules[__name__], run_dir, report["phases"]["A1"])
        save()
        idle()
        launch(BASE_IMAGE, "baseline-after")
        measure("A2")
        restore()
        reference = {(row["name"], row["draft"]): row for row in report["phases"]["A1"]["requests"]}
        comparisons = []
        for label in ("B", "A2"):
            for row in report["phases"][label]["requests"]:
                ref = reference[(row["name"], row["draft"])]
                comparisons.append({"phase": label, "name": row["name"], "draft": row["draft"],
                                    "same_output": row["output_sha256"] == ref["output_sha256"],
                                    "same_full_token_ids": bool(row.get("token_ids_sha256")) and row.get("token_ids") == ref.get("token_ids"),
                                    "same_short_token_hash": bool(row["token_sha"]) and row["token_sha"] == ref["token_sha"]})
        report["comparisons"] = comparisons
        report["quality_passed"] = all(row["same_output"] and row["same_short_token_hash"] and row["same_full_token_ids"] for row in comparisons)
        for phase in report["phases"].values():
            for row in phase["concurrent"]:
                report["quality_passed"] &= bool(row["complete"] and row["tokens"] == 128 and
                    row["same_solo_output"] and row["same_solo_short_hash"] and row["same_solo_full_ids"])
            for item in fixtures + bench.PROMPTS:
                pair = [row for row in phase["requests"] if row["name"] == item["name"]]
                report["quality_passed"] &= len(pair) == 2 and pair[0]["output_sha256"] == pair[1]["output_sha256"] and bool(pair[0].get("token_ids_sha256")) and pair[0].get("token_ids") == pair[1].get("token_ids")
        report["speed"] = {}
        for count in contexts:
            values = {}
            for label in ("A1", "B", "A2"):
                values[label] = statistics.median(row["ttft_s"] for row in report["phases"][label]["requests"]
                                                  if row["draft"] and row["name"].startswith(f"context-{count}-"))
            baseline = (values["A1"] + values["A2"]) / 2
            report["speed"][str(count)] = {"median_ttft_s": values,
                "ttft_reduction_pct": 100 * (1 - values["B"] / baseline),
                "baseline_drift_pct": 100 * (values["A2"] / values["A1"] - 1)}
        report["retrieval_accuracy"] = {}
        for label, phase in report["phases"].items():
            report["retrieval_accuracy"][label] = {}
            for count in contexts:
                flags = [ok for row in phase["requests"] if row["draft"] and row["name"].startswith(f"context-{count}-")
                         for ok in row.get("retrieval", {}).values()]
                report["retrieval_accuracy"][label][str(count)] = {"correct": sum(flags), "total": len(flags)}
        save()
        print(json.dumps({"receipt": str(path), "speed": report["speed"],
                          "retrieval_accuracy": report["retrieval_accuracy"]}), flush=True)
        if not report["quality_passed"]:
            raise RuntimeError("quality comparison failed; baseline has been restored")
    finally:
        signal.alarm(0)
        if changed:
            restore()


if __name__ == "__main__":
    main()
