"""
gym/server.py
=============
NeMo Gym FastAPI server exposing verifiable rewards for GRPO training of
ShopFloor-Nemotron.

# ---------------------------------------------------------------------------
# ANTI-REWARD-HACKING NOTES (read me before changing reward weights)
# ---------------------------------------------------------------------------
#
# Reward hackers we have seen / blocked in this codebase:
#
#   1. EMPTY-JSON HACK     — model returns "{}" or "null". JSON parses, schema
#                            "validates" loosely, reward leaks. -> We force a
#                            zero reward when assistant content is empty/null/
#                            "{}" or fails to provide at least 2 keys.
#
#   2. SCHEMA-ONLY HACK    — model emits a perfectly-shaped JSON but with
#                            bogus values (HSN "00000000", IS "IS 0000:0000").
#                            Schema match alone is capped at 0.30 of total
#                            reward; the remaining 0.70 requires content
#                            verifiers (BIS lookup, HSN top-1, T-code lookup,
#                            confidence calibration) to pass.
#
#   3. CONFIDENCE INVERSION— model emits confidence=1.0 on hallucinations.
#                            verify_confidence_calibration penalises high
#                            confidence on wrong answers (over_confident_wrong
#                            -> 0.0) and rewards calibrated refusals.
#
#   4. KEY STUFFING        — model dumps every possible key including
#                            unrelated ones to maximise schema overlap.
#                            Pydantic schemas in eval/shopbench_in_spec are
#                            strict; extra keys cause schema_error.
#
# Total reward formula:
#
#   r = 0.30 * schema_match + 0.25 * task_specific_lookup
#       + 0.25 * tcode_or_field_correctness + 0.20 * confidence_calibration
#
# Production note: this in-memory session store is NOT durable. Swap for
# Redis (see commented service in docker-compose.yaml) before multi-replica
# deployment, otherwise sessions vanish on restart.
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Import sibling eval/verifiers.py.
# Container layout: WORKDIR=/app, with /app/eval and /app/gym at the top.
# Local layout    : repo root has eval/ and gym/ — we add repo root to sys.path
#                   so `from eval.verifiers import ...` always works.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.verifiers import (  # noqa: E402
    verify_bis_is,
    verify_confidence_calibration,
    verify_hsn_top1,
    verify_rca_schema,
    verify_tcode,
    verify_task,
)

# --------------------------------------------------------------------------- #
# Pydantic v2 models
# --------------------------------------------------------------------------- #

TaskType = Literal["rca", "hsn", "bis", "sap_pm", "tcode"]


class SeedSessionRequest(BaseModel):
    task_type: TaskType | None = Field(
        default=None, description="If omitted, server picks round-robin."
    )
    seed_id: str | None = None


class InitialState(BaseModel):
    task_id: str
    task_type: TaskType
    prompt: str
    expected_output_schema: dict[str, Any]


class SeedSessionResponse(BaseModel):
    session_id: str
    initial_state: InitialState


class VerifyRequest(BaseModel):
    session_id: str
    model_output: str | dict[str, Any]


class PerComponent(BaseModel):
    schema_match: float = 0.0
    bis: float = 0.0
    hsn: float = 0.0
    tcode: float = 0.0
    conf_cal: float = 0.0


class VerifyResponse(BaseModel):
    reward: float
    terminated: bool
    info: dict[str, Any]


class ToolRequest(BaseModel):
    query: str | None = None
    code: str | None = None  # e.g. an HSN or IS string the model wants to look up
    payload: dict[str, Any] | None = None


class ToolResponse(BaseModel):
    tool: str
    ok: bool
    result: dict[str, Any]


# --------------------------------------------------------------------------- #
# Session store (in-memory; swap for Redis in prod)
# --------------------------------------------------------------------------- #

class _SessionStore:
    """Thread-safe in-memory session store.

    Production deployments must replace this with Redis — see the commented
    service block in gym/docker-compose.yaml. As-is the store is per-replica
    and sessions disappear on restart.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, payload: dict[str, Any]) -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            self._sessions[sid] = {**payload, "created_at": time.time()}
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        with self._lock:
            return self._sessions.get(sid)

    def terminate(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


SESSIONS = _SessionStore()


# --------------------------------------------------------------------------- #
# Demo task bank (replace with a real loader in prod).
# --------------------------------------------------------------------------- #

_DEMO_TASKS: list[dict[str, Any]] = [
    {
        "task_type": "rca",
        "prompt": (
            "Sir, CNC machine ka spindle baar baar trip ho raha hai shift ke beech "
            "mein, coolant bhi normal hai. Kya karna chahiye?"
        ),
        "gold": {"IS_reference": "IS 2062:2011"},
        "expected_output_schema": {
            "type": "object",
            "required": [
                "complaint_variant", "probable_cause", "corrective_action",
                "MTTR_min", "IS_reference", "HSN_code", "confidence",
            ],
        },
    },
    {
        "task_type": "hsn",
        "prompt": "M.S. angle 50x50x6 mm, IS 2062 E250, 6 meter length, 10 nos",
        "gold": {"hsn_code": "72162100"},
        "expected_output_schema": {
            "type": "object",
            "required": ["hsn_code", "gst_rate", "rationale", "confidence"],
        },
    },
    {
        "task_type": "bis",
        "prompt": "Structural steel I-beams for factory shed in Pune — which IS applies?",
        "gold": {"is_number": "IS 2062:2011"},
        "expected_output_schema": {
            "type": "object",
            "required": ["IS_reference", "scope", "refusal", "notes", "confidence"],
        },
    },
    {
        "task_type": "sap_pm",
        "prompt": (
            "Boss, palletizer ka gripper plate bend ho gaya hai, line 3 down hai "
            "2 ghante se, abhi tak fitter nahi aaya."
        ),
        "gold": {"tcode": "IW21", "priority": 2},
        "expected_output_schema": {
            "type": "object",
            "required": [
                "t_code", "notif_type", "priority", "short_text",
                "long_text", "probable_cause", "corrective_action",
                "MTTR_min", "IS_reference", "confidence",
            ],
        },
    },
]


# --------------------------------------------------------------------------- #
# Output sanitisation + reward aggregation
# --------------------------------------------------------------------------- #

REWARD_WEIGHTS = {
    "schema": 0.30,
    "task":   0.25,  # bis/hsn/tcode lookup specific to task_type
    "field":  0.25,  # extra field correctness (e.g. priority match)
    "conf":   0.20,
}
SCHEMA_ONLY_CAP = 0.30  # never exceed this if only schema passes


def _coerce_output(model_output: str | dict[str, Any]) -> dict[str, Any] | None:
    """Return a parsed dict or None for empty/garbage."""
    if model_output is None:
        return None
    if isinstance(model_output, dict):
        return model_output if model_output else None
    s = str(model_output).strip()
    if not s or s.lower() in {"null", "none", "{}", "[]"}:
        return None
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or len(parsed) < 2:
        # Guard #1: empty/near-empty JSON hack.
        return None
    return parsed


def _reward(task_type: str, output_d: dict[str, Any] | None, gold: dict[str, Any]) -> tuple[float, PerComponent, dict[str, Any]]:
    pc = PerComponent()
    info: dict[str, Any] = {}

    if output_d is None:
        info["reason"] = "empty_or_unparseable_output"
        return 0.0, pc, info

    results = verify_task(task_type, output_d, gold=gold)

    # schema component
    schema_key = next((k for k in results if k.endswith("_schema") or k.endswith("schema_match")), None)
    if schema_key:
        pc.schema_match = float(results[schema_key][1])

    # bis component
    if "bis_is_lookup" in results:
        pc.bis = float(results["bis_is_lookup"][1])

    # hsn component
    if "hsn_top1" in results:
        pc.hsn = float(results["hsn_top1"][1])

    # tcode component
    if "tcode_resolution" in results:
        pc.tcode = float(results["tcode_resolution"][1])

    # confidence calibration
    if "confidence_calibration" in results:
        pc.conf_cal = float(results["confidence_calibration"][1])

    # Aggregate. Pick the dominant task-specific component for `task`, and use
    # field-specific extras (priority/etc.) for `field` if present.
    task_component = max(pc.bis, pc.hsn, pc.tcode)
    field_component = 0.0
    if task_type == "sap_pm" and "priority" in output_d and "priority" in gold:
        field_component = 1.0 if int(output_d["priority"]) == int(gold["priority"]) else 0.0
    elif task_type == "rca":
        # reward content presence in RCA: corrective_action + MTTR_min sane
        ca = output_d.get("corrective_action", "")
        mttr = output_d.get("MTTR_min")
        field_component = 1.0 if (isinstance(ca, str) and ca and isinstance(mttr, int) and 0 < mttr <= 480) else 0.0
    elif task_type == "hsn":
        gst = output_d.get("gst_rate")
        field_component = 1.0 if gst in (0, 5, 12, 18, 28) else 0.0
    elif task_type == "bis":
        field_component = 0.0 if bool(output_d.get("refusal", False)) and pc.bis == 0.0 else 1.0 if pc.bis > 0 else 0.0

    total = (
        REWARD_WEIGHTS["schema"] * pc.schema_match
        + REWARD_WEIGHTS["task"]  * task_component
        + REWARD_WEIGHTS["field"] * field_component
        + REWARD_WEIGHTS["conf"]  * pc.conf_cal
    )

    # Guard #2: schema-only hack — cap reward when only the schema verifier passed.
    nonzero_other = (task_component > 0 or field_component > 0 or pc.conf_cal > 0)
    if pc.schema_match > 0 and not nonzero_other:
        total = min(total, SCHEMA_ONLY_CAP)
        info["capped"] = "schema_only"

    info["verifiers"] = {k: {"passed": v[0], "reward": v[1], "error": v[2]} for k, v in results.items()}
    return round(float(total), 4), pc, info


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="ShopFloor-Nemotron Gym",
    version="0.1.0",
    description="NeMo Gym FastAPI environment with verifiable rewards for GRPO.",
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "sessions": len(SESSIONS), "ts": time.time()}


@app.post("/seed_session", response_model=SeedSessionResponse)
def seed_session(req: SeedSessionRequest) -> SeedSessionResponse:
    if req.task_type is not None:
        candidates = [t for t in _DEMO_TASKS if t["task_type"] == req.task_type]
        if not candidates:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no task for type {req.task_type}")
        task = candidates[0]
    else:
        task = _DEMO_TASKS[len(SESSIONS) % len(_DEMO_TASKS)]

    task_id = req.seed_id or f"{task['task_type']}-{uuid.uuid4().hex[:8]}"
    sid = SESSIONS.create({
        "task_id": task_id,
        "task_type": task["task_type"],
        "prompt": task["prompt"],
        "gold": task["gold"],
    })
    return SeedSessionResponse(
        session_id=sid,
        initial_state=InitialState(
            task_id=task_id,
            task_type=task["task_type"],
            prompt=task["prompt"],
            expected_output_schema=task["expected_output_schema"],
        ),
    )


@app.post("/my_tool/{tool_name}", response_model=ToolResponse)
def my_tool(tool_name: str, req: ToolRequest) -> ToolResponse:
    """Read-only tool endpoints. Model uses these mid-rollout to look up codes."""
    if tool_name == "hsn_lookup":
        code = (req.code or "").strip()
        if len(code) != 8 or not code.isdigit():
            return ToolResponse(tool=tool_name, ok=False, result={"error": "hsn must be 8 digits"})
        chapter = code[:2]
        return ToolResponse(tool=tool_name, ok=True, result={
            "hsn_code": code,
            "chapter": chapter,
            "note": f"HSN chapter {chapter} — see GST tariff.",
        })

    if tool_name == "bis_is_lookup":
        is_number = (req.code or req.query or "").strip()
        passed, reward, err = verify_bis_is(is_number)
        return ToolResponse(tool=tool_name, ok=passed, result={
            "is_number": is_number,
            "reward": reward,
            "error": err,
        })

    if tool_name == "tcode_lookup":
        tcode = (req.code or "").strip().upper()
        passed, reward, err = verify_tcode(tcode)
        return ToolResponse(tool=tool_name, ok=passed, result={
            "tcode": tcode,
            "reward": reward,
            "error": err,
        })

    if tool_name == "rca_schema_check":
        payload = req.payload or {}
        passed, reward, err = verify_rca_schema(payload)
        return ToolResponse(tool=tool_name, ok=passed, result={
            "reward": reward,
            "error": err,
        })

    raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown tool {tool_name}")


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest) -> VerifyResponse:
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown session_id {req.session_id}")

    output_d = _coerce_output(req.model_output)
    reward, per, info = _reward(sess["task_type"], output_d, sess.get("gold", {}))

    # Mark terminated regardless of reward — one-shot env.
    SESSIONS.terminate(req.session_id)

    return VerifyResponse(
        reward=reward,
        terminated=True,
        info={
            "per_component": per.model_dump(),
            "task_type": sess["task_type"],
            **info,
        },
    )


# --------------------------------------------------------------------------- #
# Local dev runner
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gym.server:app", host="0.0.0.0", port=8000, reload=False)
