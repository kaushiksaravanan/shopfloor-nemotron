#!/usr/bin/env python
"""
gen_with_rotation.py
====================
Standalone driver for `data.generate_synthetic` that hot-rotates a pool of
Gemini API keys (free-tier 5-15 RPM each) when one hits 429. Patches the
OpenAIBackend's `complete` method to retry with the next key.

This is a one-shot helper for the bootstrap dataset; the real generator at
`data/generate_synthetic.py` remains unchanged in its public contract.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_synthetic import (  # noqa: E402
    _adapt_seed,
    _extract_user_turn,
    _jinja_env,
    _load_seeds,
    _expand_seeds,
    _render_prompt,
    TRAINING_SYSTEM_PROMPT,
)


# Pool of Gemini keys vended via CipherStack. Keep last-used timestamp per key
# so we honor approximate per-key RPM.
KEY_POOL: list[dict] = []


def vend_keys(n: int = 8) -> list[dict]:
    import urllib.request
    keys: dict[str, dict] = {}
    for _ in range(n * 2):  # vend more than n to dedupe
        req = urllib.request.Request(
            "https://cipherstack.kaushik.cv/api/v1/vend/gemini",
            headers={
                "Authorization": "Bearer csk_REVOKED_DO_NOT_USE_0000000000000000000000000000000000",
                "User-Agent": "shopfloor-nemotron/sft-gen",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read().decode("utf-8"))
            keys[d["key_id"]] = {"key": d["key"], "key_id": d["key_id"], "last": 0.0, "cooldown_until": 0.0}
            if len(keys) >= n:
                break
        except Exception as e:
            print(f"[vend] {e}", file=sys.stderr)
            time.sleep(0.5)
    return list(keys.values())


def pick_key(min_gap: float = 6.0) -> dict | None:
    """Pick the LRU key whose cooldown has expired and that was last used > min_gap seconds ago."""
    now = time.time()
    available = [k for k in KEY_POOL if k["cooldown_until"] <= now]
    if not available:
        return None
    available.sort(key=lambda k: k["last"])
    k = available[0]
    wait = (k["last"] + min_gap) - now
    if wait > 0:
        time.sleep(wait)
    k["last"] = time.time()
    return k


def call_gemini(model: str, system: str, user: str, *, max_retries: int = 6) -> str:
    """Direct call to Gemini native API with key rotation."""
    import urllib.request, urllib.error

    body = {
        "system_instruction": {"parts": [{"text": system}]} if system else None,
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }
    if body["system_instruction"] is None:
        body.pop("system_instruction")

    last_err = None
    for attempt in range(max_retries):
        k = pick_key()
        if k is None:
            time.sleep(2.0)
            continue
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={k['key']}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            cands = d.get("candidates") or []
            if not cands:
                last_err = f"no candidates: {d}"
                continue
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            return text
        except urllib.error.HTTPError as e:
            body_bytes = e.read()
            txt = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            if e.code == 429:
                # Put this key on a longer cooldown
                k["cooldown_until"] = time.time() + 30.0
                last_err = "429"
                continue
            if e.code in (400, 403):
                # Probably bad key — long cooldown
                k["cooldown_until"] = time.time() + 600.0
                last_err = f"{e.code}: {txt[:200]}"
                continue
            last_err = f"{e.code}: {txt[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"gemini call failed after {max_retries} retries: {last_err}")


def render_one(env, seed: dict, model: str) -> dict | None:
    adapted = _adapt_seed(seed)
    task_type = adapted.get("task_type", "rca")
    prompt = _render_prompt(env, task_type, adapted)

    if "\nUSER:" in prompt:
        sys_part, user_part = prompt.split("\nUSER:", 1)
        sys_part = sys_part.removeprefix("SYSTEM:").strip()
        user_part = user_part.strip()
    else:
        sys_part, user_part = "", prompt

    try:
        raw = call_gemini(model, sys_part, user_part)
    except Exception as e:
        print(f"[warn] {e}", file=sys.stderr)
        return None

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return None

    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": _extract_user_turn(adapted)},
            {"role": "assistant", "content": raw},
        ],
        "task_type": task_type,
        "metadata": {"teacher": model, "seed_id": str(seed.get("id", seed.get("seed_id", "")))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="gemini-2.0-flash-lite")
    ap.add_argument("--seeds", default="eval/seed_data.jsonl")
    ap.add_argument("--output", default="data/synthetic/train.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-gap", type=float, default=4.5, help="Per-key min seconds between calls")
    args = ap.parse_args()

    global KEY_POOL
    print("[i] vending Gemini keys from CipherStack...")
    KEY_POOL = vend_keys(8)
    print(f"[i] got {len(KEY_POOL)} keys")

    seeds = _load_seeds(ROOT / args.seeds)
    print(f"[i] {len(seeds)} seeds loaded")

    rng = random.Random(7)
    sampled = list(_expand_seeds(seeds, args.n, rng))
    env = _jinja_env()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up per-key min gap (effective RPM ~= 60/min_gap)
    # With 8 keys and 4.5s gap, effective rate ~= 8*(60/4.5) = 107 RPM
    # We patch pick_key's default by binding closure
    global pick_key
    _orig_pick = pick_key
    def _pick():
        return _orig_pick(min_gap=args.min_gap)
    pick_key = _pick

    t0 = time.time()
    written = failed = 0
    with out_path.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_one, env, s, args.model): i for i, s in enumerate(sampled)}
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                failed += 1
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if written % 20 == 0:
                dt = time.time() - t0
                rate = written / max(dt, 1e-6)
                print(f"[i] {written}/{args.n} written, {failed} failed, {rate:.2f} rows/s")

    print(f"[done] {written} written, {failed} failed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
