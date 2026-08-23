# Licensed under the Apache License, Version 2.0
"""Leaderboard CLI for the ShopFloor-Nemotron results ledger.

    python -m db.leaderboard                 # default: summary table
    python -m db.leaderboard top --n 10      # top 10 by overall
    python -m db.leaderboard best            # best per task
    python -m db.leaderboard show <run_id>   # full JSON of one run
    python -m db.leaderboard ingest          # vacuum runs/*.json
    python -m db.leaderboard export-md       # emit a markdown table
"""
from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from db import results as R

app = typer.Typer(add_completion=False, help="ShopFloor-Nemotron leaderboard CLI")
console = Console()


def _fmt(x):
    if x is None:
        return "-"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


@app.command()
def top(
    n: int = typer.Option(10, "--n", help="Number of rows to show."),
    kind: str = typer.Option("all", "--kind", help="'eval' | 'baseline' | 'all'."),
) -> None:
    """Top N eval/baseline rows by overall accuracy."""
    rows = R.top_eval(n=n, kind=kind)
    if not rows:
        console.print("[yellow]No eval/baseline runs yet.[/yellow]")
        return
    tbl = Table(title=f"Top {n} ({kind}) by overall accuracy")
    for col in ("run_id", "ts_local", "model", "provider", "n_examples",
                "overall", "rca", "hsn", "bis", "sap_pm"):
        tbl.add_column(col)
    for r in rows:
        tbl.add_row(
            str(r.get("run_id") or "-")[:42],
            str(r.get("ts_local") or "-")[:19],
            str(r.get("model") or "-")[:36],
            str(r.get("provider") or "-"),
            str(r.get("n_examples") or "-"),
            _fmt(r.get("overall")),
            _fmt(r.get("rca")),
            _fmt(r.get("hsn")),
            _fmt(r.get("bis")),
            _fmt(r.get("sap_pm")),
        )
    console.print(tbl)


@app.command()
def best() -> None:
    """Best run per task (across every eval/baseline)."""
    bp = R.best_per_task()
    if not bp:
        console.print("[yellow]No runs in the ledger.[/yellow]")
        return
    tbl = Table(title="Best per task")
    tbl.add_column("task"); tbl.add_column("score", justify="right")
    tbl.add_column("model"); tbl.add_column("run_id")
    for task in ("overall", "rca", "hsn", "bis", "sap_pm", "tcode"):
        if task in bp:
            m, v, rid = bp[task]
            tbl.add_row(task, f"{v:.3f}", str(m)[:32], str(rid)[:42])
    console.print(tbl)


@app.command()
def show(run_id: str) -> None:
    """Print one run's full record (JSON)."""
    row = R.get_run(run_id)
    if not row:
        console.print(f"[red]No run found with run_id={run_id}[/red]")
        raise typer.Exit(2)
    console.print_json(json.dumps(row, indent=2, default=str))


@app.command()
def ingest() -> None:
    """Vacuum runs/*.json and outputs/sft/**/metrics.json into the DB."""
    inserted = R.ingest_json_dir("runs")
    console.print(f"Ingested [bold]{len(inserted)}[/bold] run(s):")
    for rid in inserted:
        console.print(f"  + {rid}")
    R.summary()


@app.command("export-md")
def export_md() -> None:
    """Emit a Markdown table of the eval leaderboard (for the README)."""
    rows = R.top_eval(n=50, kind="all")
    if not rows:
        console.print("_(no runs)_")
        return
    lines = [
        "| run_id | model | overall | rca | hsn | bis | sap_pm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {rid} | {model} | {overall} | {rca} | {hsn} | {bis} | {sap_pm} |".format(
                rid=str(r.get("run_id") or "-"),
                model=str(r.get("model") or "-"),
                overall=_fmt(r.get("overall")),
                rca=_fmt(r.get("rca")),
                hsn=_fmt(r.get("hsn")),
                bis=_fmt(r.get("bis")),
                sap_pm=_fmt(r.get("sap_pm")),
            )
        )
    print("\n".join(lines))


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        R.summary()


if __name__ == "__main__":
    app()
