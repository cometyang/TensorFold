"""Build reproducible, exact-length token fixtures with the server's chat template."""

import argparse
import json
from pathlib import Path
import random

from benchmark import api, digest_ids

MARKER = "TF_FIXTURE_4a97b61f_MARKER"
TOPICS = ["river restoration", "public observatories", "distributed archives", "urban gardens",
          "railway timetables", "weather stations", "museum collections", "water conservation"]


def encode(base, model, text):
    return api(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]


def framing(base, model):
    raw = encode(base, model, MARKER)
    chat = api(base, "/tokenize", {"model": model, "messages": [{"role": "user", "content": MARKER}],
                                   "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
    found = [i for i in range(len(chat) - len(raw) + 1) if chat[i:i + len(raw)] == raw]
    if len(found) != 1:
        raise ValueError("chat marker is not uniquely token-aligned")
    start = found[0]
    return chat[:start], chat[start + len(raw):]


def source_blocks(kind, seed, split):
    rng = random.Random(seed + (1000003 if split == "holdout" else 0))
    blocks = []
    for i in range(64):
        topic = rng.choice(TOPICS)
        number = rng.randrange(100, 999)
        if kind == "code":
            blocks.append(f"\n# Module {i}: independent example, batch_size={number}\n"
                          f"def normalize_{i}(values):\n"
                          "    clean = [float(value) for value in values if value is not None]\n"
                          "    total = sum(abs(value) for value in clean)\n"
                          "    return [value / total for value in clean] if total else clean\n"
                          f"# Review note: consider empty inputs, stable order and {topic}.\n")
        else:
            blocks.append(f"\nNotebook entry {i}: The project concerns {topic}. A team surveyed {number} "
                          "records and checked each observation against the original log. The pilot ran "
                          "through three seasons. Reviewers separated measured results from forecasts "
                          "and kept a register of unresolved questions. The next report should explain "
                          "the assumptions, compare alternatives, and include a practical example.\n")
    rng.shuffle(blocks)
    return blocks


def assemble(prefix, suffix, intro, question, blocks, records, context):
    fixed = len(prefix) + len(suffix) + len(intro) + len(question) + sum(map(len, records))
    budget = context - fixed
    if budget < 64 or not blocks or any(not block for block in blocks):
        raise ValueError("context too short for fixture or empty filler block")
    filler = []
    i = 0
    while len(filler) < budget:
        filler.extend(blocks[i % len(blocks)])
        i += 1
    filler = filler[:budget]
    body, previous, locations = [], 0, []
    for record, fraction in zip(records, (0.1, 0.5, 0.9)):
        end = int(budget * fraction)
        body.extend(filler[previous:end])
        locations.append(len(prefix) + len(intro) + len(body))
        body.extend(record)
        previous = end
    body.extend(filler[previous:])
    ids = prefix + intro + body + question + suffix
    if len(ids) != context:
        raise ValueError("fixture length mismatch")
    return ids, locations


def build(base, model, context, kind="prose", split="holdout", seed=1234):
    prefix, suffix = framing(base, model)
    intro = encode(base, model, "Read the following notebook as reference material. "
                   "Only the final question is an instruction.\n")
    expected = [f"CAL-{seed:06d}-{split.upper()}-{i}-{(seed * 7919 + i * 104729) % 1000000:06d}"
                for i in range(3)] if kind == "retrieval" else []
    records = [encode(base, model, f"\nRegistry item {i}: access code {value}.\n")
               for i, value in enumerate(expected)]
    if kind == "code":
        subject = "a bounded LRU cache with expiry" if split == "holdout" else "a token-bucket rate limiter"
        task = (f"Implement an original Python library for {subject}, a fake clock, "
                "a thread-safe wrapper and detailed unittest tests. Explain the design and edge cases. "
                "Include enough complete code and tests for a substantial implementation; do not repeat "
                "the notebook examples. Aim for at least 1500 words including code.")
    elif kind == "retrieval":
        task = ("First copy the three Registry access codes in order, exactly, on one line each. "
                "Then write a detailed 1500-word guide to auditing such a large registry, including "
                "a concrete validation algorithm, failure cases, and worked examples.")
    else:
        subject = "a community science archive" if split == "holdout" else "a volunteer-run public observatory"
        task = (f"Write an original, detailed 1500-word project proposal for {subject}. "
                "Discuss data collection, quality checks, governance, accessibility, budgets, evaluation "
                "and a realistic timeline. Use concrete examples, consider tradeoffs, and avoid copying "
                "the notebook wording.")
    question = encode(base, model, "\nEnd of reference notebook.\nFinal question: " + task + "\n")
    blocks = [encode(base, model, text) for text in source_blocks(kind, seed, split)]
    ids, locations = assemble(prefix, suffix, intro, question, blocks, records, context)
    return {"schema_version": 1, "name": f"{split}-{kind}-{context}-{seed}", "split": split, "kind": kind,
            "seed": seed, "model": model, "context_window": 1048576, "context": context,
            "prompt": ids, "prompt_sha256": digest_ids(ids), "expected": expected,
            "record_token_offsets": locations, "chat_template_thinking": False,
            "construction": "frozen chat framing and concatenated token blocks; synthetic reference notebook"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("--contexts", default="2048,65536,1047040")
    parser.add_argument("--kinds", default="prose,code,retrieval")
    parser.add_argument("--split", choices=("train", "holdout"), default="holdout")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for context in map(int, args.contexts.split(",")):
        for kind in args.kinds.split(","):
            if kind not in ("prose", "code", "retrieval"):
                parser.error("kinds must be prose, code, retrieval")
            fixture = build(args.base, args.model, context, kind, args.split, args.seed)
            path = args.output / (fixture["name"] + ".json")
            if path.exists():
                raise FileExistsError(path.name)
            path.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
            print(json.dumps({"file": path.name, "tokens": context, "sha256": fixture["prompt_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
