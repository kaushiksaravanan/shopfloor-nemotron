#!/usr/bin/env python
"""
generate_synthetic.py
=====================
Synthetic SFT pair generator for ShopFloor-Nemotron.

Backends
--------
- ``nemo-data-designer`` : NVIDIA build.nvidia.com NeMo Data Designer API.
  Currently STUBBED — raises ``NotImplementedError`` with a pointer to the
  official docs.  Wire this up once your NGC org has the endpoint enabled.
  Docs: https://docs.nvidia.com/nemo/data-designer/latest/index.html

- ``openai`` : gpt-4o-mini (or any chat-completions teacher) — the dev
  fallback.  Reads ``OPENAI_API_KEY`` from the environment.

Seeds
-----
Reads ``eval/seed_data.jsonl`` — one seed per line, with at minimum a
``task_type`` field (``rca`` | ``hsn`` | ``bis`` | ``sap_pm``).  Each task
type routes to a different Jinja template under ``data/templates/``.

Output
------
JSONL at ``data/synthetic/train.jsonl`` in the OpenAI-chat format::

    {
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<hinglish>"},
        {"role": "assistant", "content": "<json>"}
      ],
      "task_type": "rca|hsn|bis|sap_pm",
      "metadata":  {"teacher": "...", "seed_id": "..."}
    }

CLI
---
::

    python data/generate_synthetic.py \\
        --backend openai \\
        --n 28000 \\
        --seeds eval/seed_data.jsonl \\
        --output data/synthetic/train.jsonl \\
        --teacher gpt-4o-mini \\
        --validate-first 200
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

app = typer.Typer(add_completion=False, help="Synthetic SFT data generator.")

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "data" / "templates"

TASK_TEMPLATES: dict[str, str] = {
    "rca":    "rca_prompt.jinja",
    "hsn":    "hsn_prompt.jinja",
    "bis":    "bis_prompt.jinja",
    "sap_pm": "sap_pm_prompt.jinja",
}

# Alias from seed task_type → template task_type (seed_data.jsonl uses
# "sap_pm_draft" while templates are keyed as "sap_pm").
TASK_TYPE_ALIAS: dict[str, str] = {
    "sap_pm_draft": "sap_pm",
}


def _adapt_seed(seed: dict[str, Any]) -> dict[str, Any]:
    """Flatten seed_data.jsonl rows into a flat dict the Jinja templates expect.

    The seed file nests scenario fields under ``input``; templates expect them at
    the top level. We also synthesize a few fields the templates reference but
    seeds don't carry (e.g. ``symptom_codes`` -> []).
    """
    flat: dict[str, Any] = {}
    flat.update(seed.get("input", {}) or {})
    # Preserve top-level metadata for downstream debugging.
    for k in ("id", "seed_id", "task_type"):
        if k in seed:
            flat.setdefault(k, seed[k])

    tt = seed.get("task_type", "rca")
    tt = TASK_TYPE_ALIAS.get(tt, tt)

    # --- per-task field aliasing so templates render cleanly --------------- #
    if tt == "rca":
        flat.setdefault("machine_type", flat.get("asset_id", "UNKNOWN"))
        flat.setdefault("symptom_codes", [])
    elif tt == "hsn":
        if "bom_line" not in flat and "item_description" in flat:
            flat["bom_line"] = flat["item_description"]
    elif tt == "bis":
        if "question" not in flat and "product_description" in flat:
            flat["question"] = flat["product_description"]
    elif tt == "sap_pm":
        flat.setdefault("plant", "UNKNOWN")
        flat.setdefault("work_center", flat.get("functional_location", "UNKNOWN"))
        flat.setdefault("equipment", flat.get("asset_id", "UNKNOWN"))

    flat["task_type"] = tt
    return flat

# Shared system prompt that goes into the OUTPUT training pair (not the
# teacher prompt — the teacher prompt has its own SYSTEM block embedded in
# the Jinja template).
TRAINING_SYSTEM_PROMPT = (
    "You are ShopFloor-Nemotron, a manufacturing assistant fine-tuned on Indian "
    "shop-floor data (Hinglish/Tanglish). Reply in Indian English. Output strict "
    "JSON only. Cite IS numbers with year (e.g. 'IS 2062:2011'). If uncertain, set "
    "confidence < 0.5 and the uncertain field to null."
)


# --------------------------------------------------------------------------- #
# Jinja env
# --------------------------------------------------------------------------- #

def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(default=False, default_for_string=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_prompt(env: Environment, task_type: str, seed: dict[str, Any]) -> str:
    tmpl_name = TASK_TEMPLATES.get(task_type)
    if tmpl_name is None:
        raise ValueError(f"Unknown task_type {task_type!r}; expected one of {list(TASK_TEMPLATES)}")
    tmpl = env.get_template(tmpl_name)
    return tmpl.render(seed=seed)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TeacherCall:
    """A single teacher call request."""
    task_type: str
    seed_id: str
    prompt: str
    user_turn: str  # the Hinglish text that goes into the OUTPUT user message


class Backend:
    name: str = "abstract"

    def complete(self, prompt: str, *, teacher: str) -> str:
        raise NotImplementedError


class NemoDataDesignerBackend(Backend):
    """Stub — wire up once your NGC org has NeMo Data Designer endpoints enabled.

    See: https://docs.nvidia.com/nemo/data-designer/latest/index.html
    The real implementation should:
      1. POST a `dataset_blueprint` to /v1/blueprints
      2. Poll the job endpoint until COMPLETED
      3. Download the resulting JSONL via the presigned S3 URL
    """
    name = "nemo-data-designer"

    def complete(self, prompt: str, *, teacher: str) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "NeMo Data Designer backend is stubbed. See "
            "https://docs.nvidia.com/nemo/data-designer/latest/index.html "
            "for the dataset-blueprint API and wire up the call here."
        )


class OpenAIBackend(Backend):
    """gpt-4o-mini (or any chat-completions teacher) via the openai SDK >= 1.0."""
    name = "openai"

    def __init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "openai>=1.0 not installed. `pip install openai` to use --backend openai."
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is required for --backend openai.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str, *, teacher: str) -> str:
        # The Jinja template already embeds its own SYSTEM/USER block.  We split
        # on the first 'USER:' marker so the teacher gets a proper chat shape.
        if "\nUSER:" in prompt:
            sys_part, user_part = prompt.split("\nUSER:", 1)
            sys_part = sys_part.removeprefix("SYSTEM:").strip()
            user_part = user_part.strip()
        else:
            sys_part, user_part = "", prompt

        last_err: Exception | None = None
        for attempt in range(8):
            try:
                resp = self._client.chat.completions.create(
                    model=teacher,
                    messages=[
                        {"role": "system", "content": sys_part},
                        {"role": "user", "content": user_part},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc)
                # Rate-limit / quota — back off and retry.
                if "429" in msg or "rate_limit" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                    # try to extract retry-after seconds
                    import re
                    m = re.search(r"try again in ([0-9.]+)s", msg)
                    wait = float(m.group(1)) if m else 2.0 + attempt * 1.5
                    wait = min(wait, 30.0)
                    time.sleep(wait)
                    continue
                # Non-retryable error — re-raise.
                raise
        # Out of retries
        assert last_err is not None
        raise last_err


def _make_backend(backend: str) -> Backend:
    if backend == "nemo-data-designer":
        return NemoDataDesignerBackend()
    if backend == "openai":
        return OpenAIBackend()
    raise typer.BadParameter(f"Unknown backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Seed loading + sampling
# --------------------------------------------------------------------------- #

def _load_seeds(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise typer.BadParameter(f"Seed file not found: {path}")
    seeds: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                seeds.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(f"{path}:{ln} invalid JSON: {exc}") from exc
    if not seeds:
        raise typer.BadParameter(f"Seed file is empty: {path}")
    return seeds


def _expand_seeds(seeds: list[dict[str, Any]], n: int, rng: random.Random) -> Iterable[dict[str, Any]]:
    """Yield exactly n seeds, stratified by task_type.

    Target mix (rounded to n):
        rca           : ~33%
        hsn           : ~25%
        bis           : ~25%
        sap_pm_draft  : ~17%
    """
    targets = {"rca": 0.333, "hsn": 0.25, "bis": 0.25, "sap_pm_draft": 0.167}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for s in seeds:
        by_type.setdefault(s.get("task_type", "rca"), []).append(s)

    counts: dict[str, int] = {tt: int(round(n * w)) for tt, w in targets.items()}
    # Tweak rounding so total == n
    delta = n - sum(counts.values())
    # Distribute delta to the first key cyclically
    keys = list(counts.keys())
    i = 0
    while delta != 0:
        counts[keys[i % len(keys)]] += 1 if delta > 0 else -1
        delta += -1 if delta > 0 else 1
        i += 1

    bag: list[dict[str, Any]] = []
    for tt, k in counts.items():
        pool = by_type.get(tt) or seeds
        for _ in range(k):
            bag.append(rng.choice(pool))
    rng.shuffle(bag)
    for s in bag:
        yield s


# --------------------------------------------------------------------------- #
# Core generation
# --------------------------------------------------------------------------- #

def _extract_user_turn(seed: dict[str, Any]) -> str:
    """The Hinglish/Tamil text the student model will see at inference."""
    # seed here is the ADAPTED/flattened seed (from _adapt_seed)
    for key in ("complaint", "question", "bom_line", "item_description",
                "product_description"):
        if key in seed and seed[key]:
            return str(seed[key])
    return json.dumps(seed, ensure_ascii=False)


def _one(env: Environment, backend: Backend, seed: dict[str, Any], teacher: str) -> dict[str, Any] | None:
    adapted = _adapt_seed(seed)
    task_type = adapted.get("task_type", "rca")
    prompt = _render_prompt(env, task_type, adapted)
    try:
        raw = backend.complete(prompt, teacher=teacher)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[warn] teacher call failed: {exc}", err=True)
        return None
    raw = raw.strip()
    # Strip markdown fences if a teacher slipped them in despite instructions.
    if raw.startswith("```"):
        # remove the first fence line and any trailing fence
        lines = raw.split("\n")
        # drop first line (```json or ```)
        lines = lines[1:]
        # drop trailing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # Sanity: must be parseable JSON.  If not, drop the row — curate.py will
    # catch survivors anyway, but we save spend by filtering at source.
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
        "metadata": {
            "teacher": teacher,
            "seed_id": str(seed.get("id", seed.get("seed_id", ""))),
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@app.command()
def main(
    backend: str = typer.Option("openai", "--backend", help="nemo-data-designer | openai"),
    n: int = typer.Option(28_000, "--n", help="Total examples to generate."),
    seeds: Path = typer.Option(Path("eval/seed_data.jsonl"), "--seeds", help="Seed JSONL."),
    output: Path = typer.Option(Path("data/synthetic/train.jsonl"), "--output", help="Output JSONL."),
    teacher: str = typer.Option(
        "gpt-4o-mini",
        "--teacher",
        help="Teacher model id (e.g. nemotron-4-340b-instruct or gpt-4o-mini).",
    ),
    validate_first: int = typer.Option(
        200,
        "--validate-first",
        help="Pause after this many rows for human review. 0 = no pause.",
    ),
    workers: int = typer.Option(8, "--workers", help="Concurrent teacher calls."),
    seed_rand: int = typer.Option(7, "--seed-rand", help="RNG seed for sampling."),
) -> None:
    """Generate synthetic SFT pairs from real Hinglish shop-floor seeds."""
    rng = random.Random(seed_rand)
    seeds_path = seeds if seeds.is_absolute() else REPO_ROOT / seeds
    output_path = output if output.is_absolute() else REPO_ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[i] loading seeds   : {seeds_path}")
    seed_rows = _load_seeds(seeds_path)
    typer.echo(f"[i] seed count      : {len(seed_rows)}")
    typer.echo(f"[i] backend         : {backend}")
    typer.echo(f"[i] teacher         : {teacher}")
    typer.echo(f"[i] target rows     : {n}")
    typer.echo(f"[i] output          : {output_path}")

    env = _jinja_env()
    back = _make_backend(backend)

    sampled = list(_expand_seeds(seed_rows, n, rng))
    written = 0
    failed = 0
    t0 = time.time()

    with output_path.open("w", encoding="utf-8") as out_fh, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, env, back, s, teacher): i for i, s in enumerate(sampled)}
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                failed += 1
                continue
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()
            written += 1

            if validate_first and written == validate_first:
                dt = time.time() - t0
                typer.echo(
                    f"\n[validate] wrote first {written} rows in {dt:.1f}s "
                    f"({failed} failed). Inspect {output_path} and press Enter "
                    f"to continue, or Ctrl-C to abort."
                )
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    typer.echo("[abort] user requested stop after validate batch.")
                    raise typer.Exit(code=0)

            if written % 500 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                typer.echo(f"[i] {written:>6}/{n} rows | {failed} failed | {rate:.1f} rows/s")

    dt = time.time() - t0
    typer.echo(f"\n[done] {written} rows written, {failed} failed, {dt:.1f}s.")


if __name__ == "__main__":
    app()
