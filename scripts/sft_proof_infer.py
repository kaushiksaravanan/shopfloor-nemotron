"""Inference smoke-test for the proof LoRA adapter.

Loads `outputs/sft/proof` on top of `Qwen/Qwen2.5-0.5B-Instruct` and
generates one sample. The output text does not need to be good — this
is purely a "the adapter loads, base+adapter forward pass works, the
tokenizer round-trips Devanagari" receipt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "outputs/sft/proof"
PROMPT = "बेयरिंग जाम, P3 line down"

print(f"Loading base {BASE} ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
print(f"Loading adapter from {ADAPTER} ...", flush=True)
m = PeftModel.from_pretrained(m, ADAPTER)
m.eval()
t = AutoTokenizer.from_pretrained(BASE)

inputs = t(PROMPT, return_tensors="pt")
t0 = time.time()
with torch.no_grad():
    out = m.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=t.eos_token_id,
    )
elapsed = time.time() - t0
text = t.decode(out[0], skip_special_tokens=True)

# Persist BEFORE printing — Windows cp1252 stdout chokes on Devanagari.
sample = {
    "prompt": PROMPT,
    "generation": text,
    "elapsed_s": elapsed,
    "adapter": ADAPTER,
    "base": BASE,
}
Path(ADAPTER, "inference_sample.json").write_text(
    json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(f"\n--- generation ({elapsed:.1f}s) ---\n{text}\n--- end ---")
print("OK")
