"""Emit runs/*.json records for the v2 SFT + GRPO-sim runs.

Reads:
  outputs/sft/proof_v2/metrics.json     (sft)
  outputs/sft/proof_v2/grpo_summary.json (grpo sim)

Writes:
  runs/sft-qwen2.5-0.5b-v2.json
  runs/grpo-sim-qwen2.5-0.5b.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

OUT = Path("outputs/sft/proof_v2")
RUNS = Path("runs")
RUNS.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def write_sft() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text())
    rec = {
        "run_id": "sft-qwen2.5-0.5b-v2",
        "ts": _ts(),
        "kind": "sft",
        "model": "Qwen2.5-0.5B-LoRA-r8-v2",
        "base_model": metrics.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct"),
        "provider": "local",
        "dataset_version": "proof_train_v2",
        "n_examples": metrics.get("n_examples", 250),
        "params_json": {
            "lora_rank": metrics.get("lora_rank", 8),
            "lora_alpha": metrics.get("lora_alpha", 16),
            "max_steps": metrics.get("steps"),
            "learning_rate": 2e-4,
            "grad_accum": 1,
            "micro_batch": 1,
            "max_seq_length": 384,
            "warmup_ratio": 0.03,
            "lr_scheduler": "cosine",
            "optim": "adamw_torch",
            "precision": "fp32",
        },
        "metrics_json": {
            "final_loss": metrics.get("final_loss"),
            "tokens_per_s": metrics.get("tokens_per_s"),
            "steps": metrics.get("steps"),
        },
        "artifact_path": "outputs/sft/proof_v2/",
        "elapsed_s": metrics.get("elapsed_s"),
        "notes": (
            "Extended CPU proof-run vs v1 (50 ex / 20 steps). v2 = 250 examples "
            "(50 seed + 200 synthetic teacher) for "
            f"{metrics.get('steps')} steps, LoRA r=8 on Qwen2.5-0.5B-Instruct. "
            "Production training will use NeMo AutoModel on Nemotron 3 Nano "
            "with the full 28k SDG corpus."
        ),
    }
    path = RUNS / "sft-qwen2.5-0.5b-v2.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"Wrote {path}")


def write_grpo() -> None:
    summary = json.loads((OUT / "grpo_summary.json").read_text())
    rec = {
        "run_id": "grpo-sim-qwen2.5-0.5b",
        "ts": _ts(),
        "kind": "grpo",
        "model": "Qwen2.5-0.5B-LoRA-r8-v2",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "provider": "local",
        "dataset_version": "gym-demo-tasks",
        "n_examples": summary.get("iterations", 10) * summary.get("group_size", 4),
        "params_json": {
            "iterations": summary.get("iterations"),
            "group_size": summary.get("group_size"),
            "temperature": 1.0,
            "max_new_tokens": 200,
            "gym_url": "http://127.0.0.1:8765",
            "weights_updated": False,
            "note": "Simulation only — proves gym<->model loop, not real GRPO updates.",
        },
        "metrics_json": {
            "overall_mean_reward": summary.get("overall_mean_reward"),
            "overall_max_reward": summary.get("overall_max_reward"),
            "per_iteration": summary.get("rows", []),
        },
        "artifact_path": "outputs/sft/proof_v2/",
        "elapsed_s": summary.get("elapsed_s"),
        "notes": (
            "GRPO simulation against live gym/server.py. We sample group_size "
            "completions per task, score each via /verify, and log the reward "
            "trajectory. No weight updates — production GRPO uses NeMo RL on GPU."
        ),
    }
    path = RUNS / "grpo-sim-qwen2.5-0.5b.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    write_sft()
    write_grpo()
