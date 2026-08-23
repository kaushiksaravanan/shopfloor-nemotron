"""Render the two PNG charts the deck needs.

Inputs (produced by sft_proof_v2.py and grpo_proof_sim.py):
  outputs/sft/proof_v2/loss_curve.csv
  outputs/sft/proof_v2/grpo_curve.csv

Outputs:
  outputs/sft/proof_v2/loss_curve.png
  outputs/sft/proof_v2/reward_curve.png
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/sft/proof_v2")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_loss() -> None:
    rows = _read_csv(ROOT / "loss_curve.csv")
    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.title("SFT Loss — Qwen2.5-0.5B LoRA r=8, 250 examples, CPU")
    plt.grid(True, alpha=0.3)
    out = ROOT / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def render_reward() -> None:
    rows = _read_csv(ROOT / "grpo_curve.csv")
    its = [int(r["iteration"]) for r in rows]
    mean = [float(r["mean_reward"]) for r in rows]
    mx = [float(r["max_reward"]) for r in rows]
    mn = [float(r["min_reward"]) for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(its, mean, marker="o", label="mean", linewidth=2)
    plt.plot(its, mx, marker="^", label="max", linewidth=1)
    plt.plot(its, mn, marker="v", label="min", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.title("Gym Reward — 10-iter GRPO sim, group_size=4")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = ROOT / "reward_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    render_loss()
    render_reward()
