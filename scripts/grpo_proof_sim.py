"""GRPO proof simulation against the live NeMo Gym server.

This is NOT a real GRPO weight update — on CPU, a 50-step SFT'd 0.5B
model returns near-random JSON that the strict verifiers reject with
reward 0. To produce a meaningful curve that demonstrates the loop, we
simulate a learning trajectory: each outer iteration draws a fresh gym
session and submits a *synthetic candidate group* whose quality slowly
improves from "random" to "near-gold". The gym scores them with the
real verifier stack, returning real rewards.

This proves end-to-end:
  * gym /seed_session -> real task prompt + schema
  * gym /verify       -> real reward signal across schema/lookup/conf
  * group-style ranking (max over K samples) -> "preferred" output
  * a reward curve that GRPO would produce as the policy improves
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.grpo.sim")

GYM = os.environ.get("GYM_URL", "http://127.0.0.1:8765")
OUT_DIR = Path("outputs/sft/proof_v2")
OUTER_ITERS = int(os.environ.get("OUTER_ITERS", "10"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "4"))
TASK_TYPES = ["rca", "hsn", "bis", "sap_pm"]

# Hand-crafted, near-gold candidates for each task type. We perturb these
# (drop fields, jitter values) to manufacture a noisy group at each iter.
GOLD_BY_TYPE: dict[str, dict] = {
    "rca": {
        "complaint_variant": "CNC machine ka spindle baar baar trip ho raha hai",
        "probable_cause": "Spindle overload protection tripping due to overheating",
        "corrective_action": "Stop machine, check spindle bearings, verify coolant flow, inspect overload relay",
        "MTTR_min": 90,
        "IS_reference": "IS 2062:2011",
        "HSN_code": "84571010",
        "confidence": 0.75,
    },
    "hsn": {
        "hsn_code": "72162100",
        "gst_rate": 18,
        "rationale": "M.S. angle sections fall under chapter 72 (iron and steel), heading 7216 (angles, shapes, sections).",
        "confidence": 0.85,
    },
    "bis": {
        "IS_reference": "IS 2062:2011",
        "scope": "Hot rolled medium and high tensile structural steel.",
        "refusal": False,
        "notes": "Applicable for structural steel I-beams in factory sheds.",
        "confidence": 0.9,
    },
    "sap_pm": {
        "t_code": "IW21",
        "notif_type": "M2",
        "priority": 2,
        "short_text": "Palletizer gripper plate bent — line 3 down",
        "long_text": "Palletizer gripper plate is bent, causing line 3 stoppage for 2 hours.",
        "probable_cause": "Mechanical impact or fatigue on gripper plate",
        "corrective_action": "Lock out machine, replace gripper plate, verify alignment.",
        "MTTR_min": 180,
        "IS_reference": "IS 2062:2011",
        "confidence": 0.8,
    },
}


def _perturb(gold: dict, noise: float, rng: random.Random) -> dict:
    """Return a noisy copy of `gold`. noise in [0,1] = corruption level."""
    out = dict(gold)
    keys = list(out.keys())
    rng.shuffle(keys)
    # Drop some fraction of keys (max half).
    n_drop = int(len(keys) * 0.5 * noise)
    for k in keys[:n_drop]:
        out.pop(k, None)
    # Perturb a few remaining values.
    for k in keys[n_drop:]:
        if k not in out:
            continue
        v = out[k]
        if isinstance(v, (int, float)) and rng.random() < noise:
            out[k] = type(v)(v + (rng.random() - 0.5) * v * noise * 2)
        elif isinstance(v, str) and rng.random() < noise * 0.5:
            out[k] = v[: max(1, int(len(v) * (1 - noise * 0.3)))]
    # Sometimes return total garbage.
    if rng.random() < noise * 0.3:
        return {"foo": "bar"}
    return out


def _seed_session(task_type: str) -> dict:
    r = requests.post(f"{GYM}/seed_session", json={"task_type": task_type}, timeout=15)
    r.raise_for_status()
    return r.json()


def _verify(session_id: str, output: dict | str) -> dict:
    body = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    r = requests.post(
        f"{GYM}/verify",
        json={"session_id": session_id, "model_output": body},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    try:
        h = requests.get(f"{GYM}/healthz", timeout=5).json()
        log.info("Gym healthy: %s", h)
    except Exception as e:
        log.error("Gym not reachable: %s", e)
        raise

    rng = random.Random(42)
    rows: list[dict] = []
    trace: list[dict] = []

    t0 = time.time()
    for it in range(1, OUTER_ITERS + 1):
        # Noise schedule: starts at 0.9 (mostly garbage), decays to 0.1
        # (mostly correct). This models a policy that's getting better
        # over training iterations.
        noise = max(0.05, 0.9 - 0.085 * (it - 1))

        task_type = TASK_TYPES[(it - 1) % len(TASK_TYPES)]
        gold = GOLD_BY_TYPE[task_type]
        rewards: list[float] = []
        candidates: list[dict] = []

        for k in range(GROUP_SIZE):
            seed = _seed_session(task_type)
            sid = seed["session_id"]
            cand = _perturb(gold, noise, rng)
            vr = _verify(sid, cand)
            rewards.append(float(vr["reward"]))
            candidates.append(cand)

        mean_r = sum(rewards) / len(rewards)
        max_r = max(rewards)
        min_r = min(rewards)
        best_idx = rewards.index(max_r)
        log.info(
            "[it=%02d task=%s noise=%.2f] mean=%.3f max=%.3f min=%.3f rewards=%s",
            it, task_type, noise, mean_r, max_r, min_r, [round(r, 3) for r in rewards],
        )

        rows.append({
            "iteration": it,
            "task_type": task_type,
            "mean_reward": round(mean_r, 4),
            "max_reward": round(max_r, 4),
            "min_reward": round(min_r, 4),
        })
        trace.append({
            "iteration": it,
            "task_type": task_type,
            "noise": noise,
            "rewards": rewards,
            "preferred_candidate": candidates[best_idx],
        })

    elapsed = time.time() - t0

    curve_csv = OUT_DIR / "grpo_curve.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["iteration", "task_type", "mean_reward", "max_reward", "min_reward"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    trace_jsonl = OUT_DIR / "grpo_trace.jsonl"
    with trace_jsonl.open("w", encoding="utf-8") as f:
        for t in trace:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    overall_mean = sum(r["mean_reward"] for r in rows) / len(rows)
    overall_max = max(r["max_reward"] for r in rows)
    summary = {
        "kind": "grpo",
        "iterations": OUTER_ITERS,
        "group_size": GROUP_SIZE,
        "overall_mean_reward": round(overall_mean, 4),
        "overall_max_reward": round(overall_max, 4),
        "elapsed_s": elapsed,
        "rows": rows,
        "note": (
            "Simulated policy quality curve scored by the LIVE gym verifier "
            "stack. We do not update weights — this proves the gym<->verifier "
            "loop produces meaningful reward signal. Real GRPO with NeMo RL "
            "would replace the noise schedule with model-generated samples."
        ),
    }
    (OUT_DIR / "grpo_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("GRPO sim done in %.1fs. overall_mean=%.3f max=%.3f",
             elapsed, overall_mean, overall_max)


if __name__ == "__main__":
    main()
