# Licensed under the Apache License, Version 2.0
"""Pydantic v2 schemas + JSON Schemas for the ShopFloor-IN benchmark.

This module defines the shape of every benchmark task (RCA, HSN, BIS, SAP-PM),
the expected gold output for each, and utilities to freeze/hash a benchmark
file so eval runs are reproducible across the team.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = typer.Typer(add_completion=False, help="ShopFloor-IN benchmark spec utilities.")


# ---------------------------------------------------------------------------
# Output schemas (what the model is graded against)
# ---------------------------------------------------------------------------

class RCAOutput(BaseModel):
    """Structured Root-Cause-Analysis returned by the model."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., description="Equipment / functional location identifier.")
    symptom: str = Field(..., description="One-line normalized symptom in English.")
    root_cause: str = Field(..., description="Diagnosed underlying cause.")
    corrective_action: str = Field(..., description="Recommended fix / next step.")
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sap_pm_tcode: str | None = Field(None, description="Suggested SAP-PM transaction code.")


class HSNOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hsn_code: str = Field(..., pattern=r"^\d{8}$")
    description: str
    gst_rate: float = Field(..., ge=0.0, le=28.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class BISCitationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_number: str = Field(..., description="e.g. 'IS 14543:2004'")
    title: str
    domain: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("is_number")
    @classmethod
    def _is_number_shape(cls, v: str) -> str:
        if not v.upper().startswith("IS "):
            raise ValueError("is_number must start with 'IS '")
        return v


class TCodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tcode: str = Field(..., min_length=3, max_length=8)
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class SAPPMNotificationOutput(BaseModel):
    """Full SAP-PM notification draft."""

    model_config = ConfigDict(extra="forbid")

    notification_type: Literal["M1", "M2", "M3"] = Field(..., description="M1=malfunction, M2=activity, M3=request")
    functional_location: str
    equipment_id: str | None = None
    short_text: str = Field(..., max_length=40)
    long_text: str
    priority: Literal["1-very-high", "2-high", "3-medium", "4-low"]
    breakdown_indicator: bool
    reported_by: str
    tcode: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Input task schemas
# ---------------------------------------------------------------------------

class _TaskBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str
    input: dict[str, Any]
    gold_output: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RCATask(_TaskBase):
    task_type: Literal["rca"] = "rca"


class HSNTask(_TaskBase):
    task_type: Literal["hsn"] = "hsn"


class BISCitationTask(_TaskBase):
    task_type: Literal["bis"] = "bis"


class TCodeTask(_TaskBase):
    task_type: Literal["tcode"] = "tcode"


class SAPPMTask(_TaskBase):
    task_type: Literal["sap_pm_draft"] = "sap_pm_draft"


BenchmarkTask = RCATask | HSNTask | BISCitationTask | TCodeTask | SAPPMTask


TASK_OUTPUT_SCHEMA: dict[str, type[BaseModel]] = {
    "rca": RCAOutput,
    "hsn": HSNOutput,
    "bis": BISCitationOutput,
    "tcode": TCodeOutput,
    "sap_pm_draft": SAPPMNotificationOutput,
}


def output_json_schema(task_type: str) -> dict[str, Any]:
    """Return the JSON Schema for the expected output of a task type."""
    if task_type not in TASK_OUTPUT_SCHEMA:
        raise KeyError(f"Unknown task_type: {task_type}")
    return TASK_OUTPUT_SCHEMA[task_type].model_json_schema()


# ---------------------------------------------------------------------------
# Freezing / hashing
# ---------------------------------------------------------------------------

def compute_sha256(benchmark_path: str | Path) -> str:
    """Compute deterministic SHA-256 over the benchmark file content."""
    p = Path(benchmark_path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_task(raw: dict[str, Any]) -> BenchmarkTask:
    task_type = raw.get("task_type")
    mapping: dict[str, type[_TaskBase]] = {
        "rca": RCATask,
        "hsn": HSNTask,
        "bis": BISCitationTask,
        "tcode": TCodeTask,
        "sap_pm_draft": SAPPMTask,
    }
    if task_type not in mapping:
        raise ValueError(f"Unknown task_type: {task_type!r}")
    return mapping[task_type].model_validate(raw)


def freeze_benchmark(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate every line of `input_path`, re-serialize deterministically,
    and write `output_path` plus a `<output>.sha256` sidecar.

    Returns a small manifest dict.
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[BenchmarkTask] = []
    with inp.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(_validate_task(json.loads(line)))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{inp}:{ln}: {e}") from e

    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for t in tasks:
            fh.write(json.dumps(t.model_dump(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    digest = compute_sha256(out)
    Path(str(out) + ".sha256").write_text(digest + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.task_type] = counts.get(t.task_type, 0) + 1

    return {
        "input": str(inp),
        "output": str(out),
        "sha256": digest,
        "n_tasks": len(tasks),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command("freeze")
def cli_freeze(
    input: str = typer.Option(..., "--input", help="Path to seed JSONL."),
    output: str = typer.Option(..., "--output", help="Path to frozen JSONL."),
) -> None:
    """Validate + freeze a seed benchmark file."""
    manifest = freeze_benchmark(input, output)
    typer.echo(json.dumps(manifest, indent=2))


@app.command("hash")
def cli_hash(path: str = typer.Argument(..., help="Benchmark file to hash.")) -> None:
    typer.echo(compute_sha256(path))


@app.command("schema")
def cli_schema(task_type: str = typer.Argument(..., help="rca|hsn|bis|tcode|sap_pm_draft")) -> None:
    typer.echo(json.dumps(output_json_schema(task_type), indent=2))


if __name__ == "__main__":
    app()
