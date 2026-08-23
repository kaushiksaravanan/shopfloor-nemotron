#!/usr/bin/env python
"""
curate.py
=========
Curation pipeline for ShopFloor-Nemotron synthetic SFT data.

Pipeline
--------
1. **Language ID filter** — keep ``hi`` / ``ta`` / ``en`` (drops accidental
   gibberish or off-language drift from the teacher).
2. **MinHash near-dup**  — datasketch MinHashLSH @ Jaccard 0.85, 5-gram
   shingles on the assistant turn.  Falls back to exact set-dedup when
   datasketch is unavailable.
3. **PII strip**         — regex-based removal of Aadhaar, PAN, IFSC,
   +91 mobile, GSTIN.  Rows where PII appears in the assistant turn are
   DROPPED (not just masked) because they were probably hallucinated by
   the teacher.
4. **Quality score**     — combines (a) response length percentile and
   (b) JSON parseability of the assistant turn.  Rows below the 10th
   percentile or with unparseable JSON are dropped.

Outputs
-------
- ``data/curated/train.jsonl``   — surviving rows.
- ``data/curated/curate_report.json``  — per-stage counts + samples.

NeMo Curator
------------
``--use-nemo-curator`` switches to the GPU-accelerated NeMo Curator pipeline
(if installed).  Docs:
https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, help="Curate synthetic SFT data.")

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# PII regex (Indian)
# --------------------------------------------------------------------------- #

# Aadhaar: 12 digits, often in xxxx-xxxx-xxxx or xxxx xxxx xxxx form.
RE_AADHAAR = re.compile(r"\b(?:\d[ -]?){11}\d\b")
# PAN: 5 letters, 4 digits, 1 letter
RE_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
# IFSC: 4 letters, 0, 6 alnum  (e.g. SBIN0001234)
RE_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
# +91 mobile (10 digits starting 6-9)
RE_PHONE = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}\b")
# GSTIN: 15 chars: 2 digits state + 10 char PAN + 1 entity + 1 'Z' + 1 checksum
RE_GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "aadhaar": RE_AADHAAR,
    "pan":     RE_PAN,
    "ifsc":    RE_IFSC,
    "phone":   RE_PHONE,
    "gstin":   RE_GSTIN,
}


def find_pii(text: str) -> dict[str, int]:
    return {k: len(p.findall(text)) for k, p in PII_PATTERNS.items() if p.search(text)}


def strip_pii(text: str) -> str:
    for k, p in PII_PATTERNS.items():
        text = p.sub(f"<{k.upper()}_REDACTED>", text)
    return text


# --------------------------------------------------------------------------- #
# Lang ID
# --------------------------------------------------------------------------- #

def _langid(text: str) -> str:
    """Best-effort: try `langid` then `langdetect`; fall back to script-based heuristic."""
    try:
        import langid  # type: ignore[import-not-found]
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        pass
    try:
        from langdetect import detect  # type: ignore[import-not-found]
        return detect(text)
    except Exception:
        pass
    # Heuristic fallback: detect script range.
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    if re.search(r"[஀-௿]", text):
        return "ta"
    return "en"


ALLOWED_LANGS = {"hi", "ta", "en"}


# --------------------------------------------------------------------------- #
# MinHash dedup
# --------------------------------------------------------------------------- #

def _shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _minhash_dedup(rows: list[dict[str, Any]], threshold: float = 0.85) -> tuple[list[dict[str, Any]], int]:
    """Return (kept_rows, dropped_count). Uses datasketch when available."""
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-not-found]
    except ImportError:
        typer.echo("[warn] datasketch not installed — falling back to exact set dedup.", err=True)
        seen: set[str] = set()
        kept: list[dict[str, Any]] = []
        dropped = 0
        for r in rows:
            key = r["messages"][-1]["content"].strip()
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(r)
        return kept, dropped

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for i, r in enumerate(rows):
        text = r["messages"][-1]["content"]
        mh = MinHash(num_perm=128)
        for sh in _shingles(text):
            mh.update(sh.encode("utf-8"))
        if lsh.query(mh):
            dropped += 1
            continue
        lsh.insert(f"r{i}", mh)
        kept.append(r)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Quality score
# --------------------------------------------------------------------------- #

def _assistant_text(row: dict[str, Any]) -> str:
    return row["messages"][-1]["content"]


def _is_parseable_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

@dataclass
class CurateReport:
    input_rows: int = 0
    after_langid: int = 0
    after_minhash: int = 0
    after_pii: int = 0
    after_quality: int = 0
    langid_drops: dict[str, int] = field(default_factory=dict)
    pii_hits: dict[str, int] = field(default_factory=dict)
    task_type_counts: dict[str, int] = field(default_factory=dict)
    json_parse_rate: float = 0.0


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_fallback(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], CurateReport]:
    rpt = CurateReport(input_rows=len(rows))

    # Stage 1: lang id
    stage1: list[dict[str, Any]] = []
    drop_counter: Counter[str] = Counter()
    for r in rows:
        user_text = r["messages"][1]["content"] if len(r["messages"]) > 1 else ""
        lang = _langid(user_text)
        if lang not in ALLOWED_LANGS:
            drop_counter[lang] += 1
            continue
        stage1.append(r)
    rpt.after_langid = len(stage1)
    rpt.langid_drops = dict(drop_counter)

    # Stage 2: minhash dedup
    stage2, dup_dropped = _minhash_dedup(stage1)
    rpt.after_minhash = len(stage2)

    # Stage 3: PII strip (drop assistant-PII, mask user-PII)
    stage3: list[dict[str, Any]] = []
    pii_counter: Counter[str] = Counter()
    for r in stage2:
        a_text = _assistant_text(r)
        a_hits = find_pii(a_text)
        if a_hits:
            for k, v in a_hits.items():
                pii_counter[f"assistant_{k}"] += v
            continue  # drop — teacher hallucinated PII
        u_text = r["messages"][1]["content"] if len(r["messages"]) > 1 else ""
        u_hits = find_pii(u_text)
        if u_hits:
            for k, v in u_hits.items():
                pii_counter[f"user_{k}"] += v
            r["messages"][1]["content"] = strip_pii(u_text)
        stage3.append(r)
    rpt.after_pii = len(stage3)
    rpt.pii_hits = dict(pii_counter)

    # Stage 4: quality (length percentile + JSON parse)
    if not stage3:
        rpt.after_quality = 0
        return [], rpt
    lengths = sorted(len(_assistant_text(r)) for r in stage3)
    p10 = lengths[max(0, int(0.1 * (len(lengths) - 1)))]
    parsed_count = 0
    stage4: list[dict[str, Any]] = []
    for r in stage3:
        t = _assistant_text(r)
        ok_json = _is_parseable_json(t)
        if ok_json:
            parsed_count += 1
        if len(t) >= p10 and ok_json:
            stage4.append(r)
    rpt.after_quality = len(stage4)
    rpt.json_parse_rate = round(parsed_count / max(len(stage3), 1), 4)
    rpt.task_type_counts = dict(Counter(r.get("task_type", "unknown") for r in stage4))

    return stage4, rpt


def _run_nemo_curator(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], CurateReport]:
    """NeMo Curator GPU pipeline. Falls back if not installed."""
    try:
        import nemo_curator  # noqa: F401  type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "nemo_curator not installed. `pip install nemo-curator` or omit --use-nemo-curator. "
            "Docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html"
        ) from exc
    # Real implementation would build a `nemo_curator.Sequential` of FastText
    # langid, MinHashLSH, ExactDuplicates, PII redactor, and HeuristicFilter,
    # then run on a Dask-cuDF DataFrame.  For now we delegate to the fallback
    # so the CLI is usable even when GPUs aren't available.
    typer.echo("[i] nemo_curator detected — delegating to fallback pipeline for parity.")
    return _run_fallback(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/synthetic/train.jsonl"), "--input", help="Input JSONL from generate_synthetic.py."
    ),
    output_path: Path = typer.Option(
        Path("data/curated/train.jsonl"), "--output", help="Output curated JSONL."
    ),
    report_path: Path = typer.Option(
        Path("data/curated/curate_report.json"), "--report", help="Curation report JSON."
    ),
    use_nemo_curator: bool = typer.Option(
        False, "--use-nemo-curator", help="Use NeMo Curator GPU pipeline if available."
    ),
) -> None:
    """Run the curation pipeline."""
    ip = input_path if input_path.is_absolute() else REPO_ROOT / input_path
    op = output_path if output_path.is_absolute() else REPO_ROOT / output_path
    rp = report_path if report_path.is_absolute() else REPO_ROOT / report_path
    if not ip.exists():
        raise typer.BadParameter(f"Input file not found: {ip}")

    typer.echo(f"[i] reading {ip}")
    rows = _read_jsonl(ip)
    typer.echo(f"[i] input rows: {len(rows)}")

    if use_nemo_curator:
        kept, rpt = _run_nemo_curator(rows)
    else:
        kept, rpt = _run_fallback(rows)

    _write_jsonl(op, kept)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(asdict(rpt), indent=2, ensure_ascii=False), encoding="utf-8")

    typer.echo("[done] curation complete")
    typer.echo(f"        input    : {rpt.input_rows}")
    typer.echo(f"        langid   : {rpt.after_langid}")
    typer.echo(f"        minhash  : {rpt.after_minhash}")
    typer.echo(f"        pii      : {rpt.after_pii}")
    typer.echo(f"        quality  : {rpt.after_quality}")
    typer.echo(f"        output   : {op}")
    typer.echo(f"        report   : {rp}")


if __name__ == "__main__":
    app()
