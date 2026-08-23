"""Build a richer SFT proof dataset (v2).

Combines the 50 seed-derived pairs with 200 synthetic teacher-generated
pairs, sampled proportionally across task types. Normalises the
synthetic `messages` schema down to the `prompt`/`response` shape used
by `train.sft._format_example`.
"""
from __future__ import annotations
import json
import pathlib
from collections import defaultdict

SEED = pathlib.Path("data/curated/proof_train.jsonl")
SYNTH = pathlib.Path("data/synthetic/train.jsonl")
DST = pathlib.Path("data/curated/proof_train_v2.jsonl")
DST.parent.mkdir(parents=True, exist_ok=True)

out: list[dict] = []

# 1) Reuse all 50 seed-derived pairs verbatim.
with SEED.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            out.append(json.loads(line))

# 2) Convert synthetic messages -> prompt/response and bucket by task_type.
by_type: dict[str, list[dict]] = defaultdict(list)
with SYNTH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        msgs = ex.get("messages", [])
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not user or not asst:
            continue
        ttype = ex.get("task_type", "rca")
        by_type[ttype].append({"prompt": user, "response": asst})

# 3) Sample evenly across task types -> ~200 total.
TARGET = 200
n_each = TARGET // max(len(by_type), 1)
for t, items in by_type.items():
    out.extend(items[:n_each])

with DST.open("w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(out)} examples to {DST}")
print(f"  seed: 50, synth per type: {n_each}, types: {sorted(by_type)}")
