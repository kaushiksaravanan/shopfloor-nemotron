"""Wrap an eval-report JSON in the canonical runs-DB schema and re-emit in place.

Usage:
    python scripts/wrap_baseline_result.py \
        --path runs/baseline-<slug>.json \
        --slug <slug> \
        --model <full-model-id> \
        --provider <groq|mistral|cohere|nvidia> \
        --notes "..."

Reads the existing eval-report JSON, augments it with run_id, ts, kind,
provider, dataset_version, dataset_sha256, params_json, metrics_json,
artifact_path so the SQLite ledger ingester can consume it uniformly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import time
from pathlib import Path


def _short_hash(s: str, n: int = 6) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--notes", default="")
    ap.add_argument("--dataset-sha", default="")
    ap.add_argument("--dataset-version", default="shopbench-in-seed-v1")
    args = ap.parse_args()

    path = Path(args.path)
    report = json.loads(path.read_text(encoding="utf-8"))

    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    run_id = f"baseline-{args.slug}-{today}-{_short_hash(args.slug + today)}"

    per_task = report.get("per_task_accuracy", {}) or {}
    metrics_json = {
        "overall": report.get("overall_accuracy", 0.0),
        "rca": per_task.get("rca"),
        "hsn": per_task.get("hsn"),
        "bis": per_task.get("bis"),
        "sap_pm": per_task.get("sap_pm_draft"),
        "tcode": per_task.get("tcode"),
        "per_task": per_task,
        "per_signal_mean_reward": report.get("per_signal_mean_reward", {}),
        "overall_mean_reward": report.get("overall_mean_reward", 0.0),
    }

    augmented = {
        **report,
        "run_id": run_id,
        "ts": int(time.time()),
        "kind": "baseline",
        "model": args.model,
        "provider": args.provider,
        "base_model": args.base_model,
        "dataset_version": args.dataset_version,
        "dataset_sha256": args.dataset_sha,
        "n_examples": report.get("n_tasks", 0),
        "params_json": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "metrics_json": metrics_json,
        "artifact_path": str(path).replace("\\", "/"),
        "elapsed_s": report.get("wall_time_s", 0.0),
        "notes": args.notes,
    }
    path.write_text(json.dumps(augmented, indent=2), encoding="utf-8")
    print(f"Wrapped {path} as run_id={run_id}")


if __name__ == "__main__":
    main()
