# Licensed under the Apache License, Version 2.0
"""SQLite results ledger for ShopFloor-Nemotron.

Every eval / SFT / GRPO / quant / data-gen run lands here. The same module
is imported by `eval/evaluate.py` and `train/sft.py` so the ledger stays in
sync without anyone copying numbers by hand.

Schema lives in `db/schema.sql`. The DB file is `db/results.sqlite`
(gitignored — keep run JSON files under `runs/` as the durable source).
"""
from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

_DB_DIR = Path(__file__).resolve().parent
_DB_PATH = _DB_DIR / "results.sqlite"
_SCHEMA_PATH = _DB_DIR / "schema.sql"


# --------------------------------------------------------------------------- #
# Connection helpers
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> Path:
    """Create db/results.sqlite if missing and apply the schema."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with _connect() as conn:
        conn.executescript(schema)
    return _DB_PATH


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_DB_DIR.parent,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "no-git"
    except Exception:
        pass
    return "no-git"


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #
def log_run(
    kind: str,
    model: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    *,
    base_model: str | None = None,
    provider: str | None = None,
    dataset_version: str | None = None,
    dataset_sha256: str | None = None,
    n_examples: int | None = None,
    artifact_path: str | None = None,
    elapsed_s: float | None = None,
    status: str = "completed",
    notes: str | None = None,
    run_id: str | None = None,
    ts: int | None = None,
) -> str:
    """Atomically insert one row. Returns the run_id."""
    init_db()
    ts = int(ts if ts is not None else time.time())
    if run_id is None:
        safe_model = (model or "model").replace("/", "-").replace(" ", "-")
        h = hashlib.sha1(
            json.dumps({"k": kind, "m": model, "p": params, "t": ts}, sort_keys=True, default=str).encode()
        ).hexdigest()[:6]
        run_id = f"{kind}-{safe_model}-{ts}-{h}"

    payload = (
        run_id,
        ts,
        kind,
        model,
        base_model,
        provider,
        dataset_version,
        dataset_sha256,
        n_examples,
        json.dumps(params, default=str, ensure_ascii=False),
        json.dumps(metrics, default=str, ensure_ascii=False),
        artifact_path,
        _git_commit(),
        socket.gethostname(),
        elapsed_s,
        status,
        notes,
    )
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, ts, kind, model, base_model, provider,
                dataset_version, dataset_sha256, n_examples,
                params_json, metrics_json, artifact_path,
                git_commit, host, elapsed_s, status, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            payload,
        )
    return run_id


# --------------------------------------------------------------------------- #
# JSON vacuum — convert legacy runs/*.json + outputs/sft/**/metrics.json
# --------------------------------------------------------------------------- #
def _normalize_eval_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Reshape a runs/*.json eval payload into the canonical metrics shape."""
    pt = raw.get("per_task_accuracy", {}) or {}
    metrics: dict[str, Any] = {
        "overall": raw.get("overall_accuracy"),
        "overall_mean_reward": raw.get("overall_mean_reward"),
        "rca": pt.get("rca"),
        "hsn": pt.get("hsn"),
        "bis": pt.get("bis"),
        "sap_pm": pt.get("sap_pm_draft") or pt.get("sap_pm"),
        "tcode": pt.get("tcode"),
        "wall_time_s": raw.get("wall_time_s"),
        "per_signal_mean_reward": raw.get("per_signal_mean_reward", {}),
    }
    return {k: v for k, v in metrics.items() if v is not None}


def _ingest_eval_json(path: Path) -> tuple[str, str] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "per_task_accuracy" not in raw and "overall_accuracy" not in raw:
        return None
    model = raw.get("model") or "unknown"
    is_baseline = bool(raw.get("baseline_model_full_name")) or "baseline" in path.stem.lower()
    kind = "baseline" if is_baseline else "eval"
    metrics = _normalize_eval_metrics(raw)
    params = {
        "benchmark": raw.get("benchmark") or raw.get("benchmark_file"),
        "dry_run": raw.get("dry_run", False),
        "provider_endpoint": raw.get("provider_endpoint"),
        "n_tasks_by_type": raw.get("n_tasks_by_type"),
    }
    run_id = raw.get("run_id") or path.stem
    measured_at = raw.get("measured_at_utc")
    ts = None
    if measured_at:
        try:
            from datetime import datetime
            ts = int(datetime.fromisoformat(measured_at.replace("Z", "+00:00")).timestamp())
        except Exception:
            ts = None
    if ts is None:
        ts = int(path.stat().st_mtime)
    return run_id, log_run(
        kind=kind,
        model=model,
        params=params,
        metrics=metrics,
        provider=raw.get("provider"),
        n_examples=raw.get("n_tasks"),
        artifact_path=str(path),
        elapsed_s=raw.get("wall_time_s"),
        notes=raw.get("notes"),
        run_id=run_id,
        ts=ts,
    )


def _ingest_sft_metrics(path: Path) -> tuple[str, str] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "train_loss" not in raw and "backend" not in raw:
        return None
    model = raw.get("model") or "unknown-sft"
    metrics = {
        "final_loss": raw.get("train_loss"),
        "tokens_per_s": raw.get("tokens_per_s"),
        "peak_mem_gb": raw.get("peak_mem_gb"),
        "steps": raw.get("steps"),
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}
    params = {
        "lora_rank": raw.get("rank"),
        "lora_alpha": raw.get("alpha"),
        "epochs": raw.get("epochs"),
        "backend": raw.get("backend"),
    }
    run_id = raw.get("run_id") or f"sft-{model.replace('/', '-')}-{int(path.stat().st_mtime)}"
    ts = int(path.stat().st_mtime)
    return run_id, log_run(
        kind="sft",
        model=model,
        base_model=raw.get("base_model") or model,
        params=params,
        metrics=metrics,
        artifact_path=str(path.parent),
        elapsed_s=raw.get("elapsed_s"),
        run_id=run_id,
        ts=ts,
    )


def ingest_json_dir(dir_path: str | Path = "runs") -> list[str]:
    """Idempotently ingest legacy run JSON files into the DB.

    Looks at:
      - <dir_path>/*.json                (eval/baseline reports)
      - outputs/sft/**/metrics.json      (SFT proof / training reports)
    Skips run_ids that already exist.
    """
    init_db()
    project = _DB_DIR.parent
    runs_dir = (project / dir_path) if not Path(dir_path).is_absolute() else Path(dir_path)
    candidates: list[Path] = []
    if runs_dir.exists():
        candidates.extend(sorted(runs_dir.glob("*.json")))
    sft_dir = project / "outputs" / "sft"
    if sft_dir.exists():
        candidates.extend(sorted(sft_dir.rglob("metrics.json")))

    inserted: list[str] = []
    with _connect() as conn:
        existing = {r[0] for r in conn.execute("SELECT run_id FROM runs").fetchall()}
    for p in candidates:
        result: tuple[str, str] | None = None
        if p.name == "metrics.json":
            tentative_id = f"sft-{p.parent.name}"
            # Probe a stable id without inserting first
            result = _ingest_sft_metrics(p)
        else:
            result = _ingest_eval_json(p)
        if result:
            rid, _ = result
            if rid not in existing:
                inserted.append(rid)
                existing.add(rid)
    return inserted


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #
def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for k in ("params_json", "metrics_json"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                pass
    return d


def top_eval(n: int = 10, kind: str = "eval") -> list[dict[str, Any]]:
    """Top N runs by overall accuracy (kind='eval' or 'baseline' or 'all')."""
    init_db()
    with _connect() as conn:
        if kind == "all":
            rows = conn.execute(
                "SELECT * FROM leaderboard_eval LIMIT ?", (n,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM leaderboard_eval WHERE run_id IN "
                "(SELECT run_id FROM runs WHERE kind = ?) LIMIT ?",
                (kind, n),
            ).fetchall()
    return [dict(r) for r in rows]


def best_per_task() -> dict[str, tuple[str, float, str]]:
    """Return {task: (model, score, run_id)} across every eval/baseline run."""
    init_db()
    out: dict[str, tuple[str, float, str]] = {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, model, metrics_json FROM runs WHERE kind IN ('eval','baseline')"
        ).fetchall()
    for r in rows:
        m = {}
        try:
            m = json.loads(r["metrics_json"])
        except Exception:
            continue
        for task in ("rca", "hsn", "bis", "sap_pm", "tcode", "overall"):
            val = m.get(task)
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            prev = out.get(task)
            if prev is None or v > prev[1]:
                out[task] = (r["model"], v, r["run_id"])
    return out


def lineage(run_id: str) -> list[dict[str, Any]]:
    """Return runs that share dataset_version with the given run."""
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return []
        dv = row["dataset_version"]
        if not dv:
            return [_row_to_dict(row)]
        rows = conn.execute(
            "SELECT * FROM runs WHERE dataset_version = ? ORDER BY ts ASC", (dv,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_dict(row) if row else None


def counts_by_kind() -> dict[str, int]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT kind, COUNT(*) AS n FROM runs GROUP BY kind ORDER BY n DESC"
        ).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def summary() -> None:
    """Print a rich-formatted leaderboard + per-kind counts."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    init_db()
    cnt = counts_by_kind()
    if not cnt:
        console.print("[yellow]No runs in the ledger yet — run `make db-init`.[/yellow]")
        return

    ck = Table(title="Runs by kind")
    ck.add_column("kind"); ck.add_column("count", justify="right")
    total = 0
    for k, v in cnt.items():
        ck.add_row(k, str(v)); total += v
    ck.add_row("[bold]total[/bold]", f"[bold]{total}[/bold]")
    console.print(ck)

    rows = top_eval(n=10, kind="all")
    if rows:
        lb = Table(title="Top 10 eval / baseline by overall accuracy")
        for col in ("run_id", "ts_local", "model", "overall", "rca", "hsn", "bis", "sap_pm"):
            lb.add_column(col)
        for r in rows:
            def _fmt(x):
                if x is None: return "-"
                try: return f"{float(x):.3f}"
                except Exception: return str(x)
            lb.add_row(
                str(r.get("run_id"))[:42],
                str(r.get("ts_local") or "-")[:19],
                str(r.get("model") or "-")[:36],
                _fmt(r.get("overall")),
                _fmt(r.get("rca")),
                _fmt(r.get("hsn")),
                _fmt(r.get("bis")),
                _fmt(r.get("sap_pm")),
            )
        console.print(lb)

    best = best_per_task()
    if best:
        bt = Table(title="Best per task")
        bt.add_column("task"); bt.add_column("score", justify="right")
        bt.add_column("model"); bt.add_column("run_id")
        for task in ("overall", "rca", "hsn", "bis", "sap_pm", "tcode"):
            if task in best:
                m, v, rid = best[task]
                bt.add_row(task, f"{v:.3f}", str(m)[:32], str(rid)[:42])
        console.print(bt)


__all__ = [
    "init_db",
    "log_run",
    "ingest_json_dir",
    "top_eval",
    "best_per_task",
    "lineage",
    "get_run",
    "counts_by_kind",
    "summary",
]
