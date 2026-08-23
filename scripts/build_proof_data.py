"""Build a tiny SFT dataset for the CPU proof-run.

Reads `eval/seed_data.jsonl` (hand-curated RCA pairs) and emits
`data/curated/proof_train.jsonl` in the prompt/response shape expected
by `train.sft._format_example`.
"""
from __future__ import annotations
import json
import pathlib

SRC = pathlib.Path("eval/seed_data.jsonl")
DST = pathlib.Path("data/curated/proof_train.jsonl")
DST.parent.mkdir(parents=True, exist_ok=True)

rows = []
with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        prompt = json.dumps(ex["input"], ensure_ascii=False)
        response = json.dumps(ex["gold_output"], ensure_ascii=False)
        rows.append({"prompt": prompt, "response": response})

# Cap at 50 for the proof run
rows = rows[:50]

with DST.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows to {DST}")
