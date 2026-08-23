# Licensed under the Apache License, Version 2.0
"""Pure-function verifiers used by both eval and the RL gym env.

Every verifier returns `(passed: bool, reward: float in [0,1], error: str)`.
No global state; safe to call from worker processes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eval.shopbench_in_spec import (
    BISCitationOutput,
    HSNOutput,
    RCAOutput,
    SAPPMNotificationOutput,
    TCodeOutput,
)

# ---------------------------------------------------------------------------
# Lookup tables (CSV-backed, with inline fallback for offline use)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_BIS_FALLBACK: dict[str, dict[str, str]] = {
    "IS 14543:2004": {"title": "Packaged Drinking Water", "domain": "food"},
    "IS 13428:2005": {"title": "Packaged Natural Mineral Water", "domain": "food"},
    "IS 11196:1985": {"title": "Compressed Natural Gas (CNG) Cylinder", "domain": "chemicals"},
    "IS 9000:2006": {"title": "Quality Management Systems Guidelines", "domain": "quality"},
    "IS 16046:2018": {"title": "Secondary Cells and Batteries", "domain": "electrical"},
    "IS 4985:2000": {"title": "Unplasticized PVC Pipes", "domain": "manufacturing"},
    "IS 277:2018": {"title": "Galvanized Steel Sheets", "domain": "manufacturing"},
    "IS 1786:2008": {"title": "High Strength Deformed Steel Bars", "domain": "manufacturing"},
    "IS 269:2015": {"title": "Ordinary Portland Cement", "domain": "manufacturing"},
    "IS 875:1987": {"title": "Code of Practice for Design Loads", "domain": "safety"},
    "IS 694:2010": {"title": "PVC Insulated Cables", "domain": "electrical"},
    "IS 732:2019": {"title": "Electrical Wiring Installations", "domain": "electrical"},
    "IS 3043:2018": {"title": "Code of Practice for Earthing", "domain": "electrical"},
    "IS 2062:2011": {"title": "Hot Rolled Structural Steel", "domain": "manufacturing"},
    "IS 800:2007": {"title": "General Construction in Steel", "domain": "manufacturing"},
}

_TCODE_FALLBACK: dict[str, str] = {
    "IW21": "Create PM Notification",
    "IW22": "Change PM Notification",
    "IW23": "Display PM Notification",
    "IW24": "Create PM Notification (Quick Entry)",
    "IW28": "Change PM Notifications (List Editing)",
    "IW29": "Display PM Notifications (List Editing)",
    "IW31": "Create Maintenance Order",
    "IW32": "Change Maintenance Order",
    "IW33": "Display Maintenance Order",
    "IW38": "Change Maintenance Orders (List Editing)",
    "IW39": "Display Maintenance Orders (List Editing)",
    "IW41": "Confirmation of Maintenance Order",
    "IK11": "Create Measurement Document",
    "IH01": "Functional Location Structure",
    "IH08": "Display Equipment (List Editing)",
}


def _load_csv(name: str) -> list[dict[str, str]]:
    p = _DATA_DIR / name
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _bis_index() -> dict[str, dict[str, str]]:
    rows = _load_csv("bis_is_master.csv")
    if not rows:
        return _BIS_FALLBACK
    return {r["is_number"]: {"title": r["title"], "domain": r["domain"]} for r in rows}


def _tcode_index() -> dict[str, str]:
    rows = _load_csv("sap_pm_tcodes.csv")
    if not rows:
        return _TCODE_FALLBACK
    return {r["tcode"]: r["description"] for r in rows}


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

VerifierResult = tuple[bool, float, str]


def verify_json_parseable(raw: str | dict[str, Any]) -> VerifierResult:
    if isinstance(raw, dict):
        return True, 1.0, ""
    try:
        json.loads(raw)
        return True, 1.0, ""
    except (json.JSONDecodeError, TypeError) as e:
        return False, 0.0, f"json_parse_error: {e}"


def verify_rca_schema(output: str | dict[str, Any]) -> VerifierResult:
    """Validate output JSON against `RCAOutput`."""
    if isinstance(output, str):
        ok, _, err = verify_json_parseable(output)
        if not ok:
            return False, 0.0, err
        output = json.loads(output)
    try:
        RCAOutput.model_validate(output)
        return True, 1.0, ""
    except ValidationError as e:
        # Partial credit for "JSON parsed but failed schema"
        return False, 0.2, f"schema_error: {e.error_count()} issues"


def verify_bis_is(is_number: str) -> VerifierResult:
    if not isinstance(is_number, str):
        return False, 0.0, "is_number must be str"
    idx = _bis_index()
    if is_number in idx:
        return True, 1.0, ""
    # tolerate minor formatting drift: 'IS  14543:2004' -> 'IS 14543:2004'
    normalized = " ".join(is_number.split())
    if normalized in idx:
        return True, 0.9, "normalized_match"
    return False, 0.0, f"unknown_is_number: {is_number}"


def verify_hsn_top1(predicted: str, gold: str) -> VerifierResult:
    if not isinstance(predicted, str) or not isinstance(gold, str):
        return False, 0.0, "hsn must be str"
    if len(predicted) != 8 or not predicted.isdigit():
        return False, 0.0, "predicted must be 8-digit numeric"
    if predicted == gold:
        return True, 1.0, ""
    # Partial credit if first 6 digits match (HSN chapter+heading+subheading)
    if predicted[:6] == gold[:6]:
        return False, 0.5, "subheading_match_only"
    if predicted[:4] == gold[:4]:
        return False, 0.2, "heading_match_only"
    return False, 0.0, f"mismatch: pred={predicted} gold={gold}"


def verify_tcode(tcode: str) -> VerifierResult:
    if not isinstance(tcode, str):
        return False, 0.0, "tcode must be str"
    idx = _tcode_index()
    t = tcode.strip().upper()
    if t in idx:
        return True, 1.0, ""
    return False, 0.0, f"unknown_tcode: {tcode}"


def verify_confidence_calibration(
    output: str | dict[str, Any],
    answer_correct: bool | None = None,
    threshold: float = 0.6,
) -> VerifierResult:
    """Reward calibrated confidence.

    Calibrated cases (reward 1.0):
      - confidence >= threshold AND answer is correct
      - confidence <  threshold AND answer is incorrect (i.e. model refused)
    Mis-calibrated cases (reward 0.0):
      - high confidence on a wrong answer
      - low  confidence on a correct answer (under-confidence is mildly penalized)
    If `answer_correct` is None we only check that the field exists & is in-range.
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as e:
            return False, 0.0, f"json_parse_error: {e}"
    if not isinstance(output, dict) or "confidence" not in output:
        return False, 0.0, "missing confidence field"

    conf = output["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        return False, 0.0, f"confidence out of range: {conf}"
    conf = float(conf)

    if answer_correct is None:
        return True, 1.0, ""

    high = conf >= threshold
    if high and answer_correct:
        return True, 1.0, "high_conf_correct"
    if (not high) and (not answer_correct):
        return True, 1.0, "low_conf_refused"
    if high and not answer_correct:
        return False, 0.0, "over_confident_wrong"
    return False, 0.5, "under_confident_correct"


# ---------------------------------------------------------------------------
# Convenience: route a (task_type, output, gold) triple to the right verifier
# ---------------------------------------------------------------------------

def verify_task(
    task_type: str,
    output: str | dict[str, Any],
    gold: dict[str, Any] | None = None,
) -> dict[str, VerifierResult]:
    """Run the appropriate verifiers for a task type. Returns a dict of named
    verifier results so callers can aggregate per-reward-signal."""
    if isinstance(output, str):
        try:
            output_d = json.loads(output)
        except json.JSONDecodeError as e:
            return {"json": (False, 0.0, f"json_parse_error: {e}")}
    else:
        output_d = output

    results: dict[str, VerifierResult] = {"json": (True, 1.0, "")}

    if task_type == "rca":
        results["rca_schema_match"] = verify_rca_schema(output_d)
        if gold and "sap_pm_tcode" in output_d and output_d["sap_pm_tcode"]:
            results["tcode_resolution"] = verify_tcode(output_d["sap_pm_tcode"])
    elif task_type == "hsn":
        try:
            HSNOutput.model_validate(output_d)
            results["hsn_schema"] = (True, 1.0, "")
        except ValidationError as e:
            results["hsn_schema"] = (False, 0.0, str(e.error_count()))
        if gold and "hsn_code" in gold:
            results["hsn_top1"] = verify_hsn_top1(
                output_d.get("hsn_code", ""), gold["hsn_code"]
            )
    elif task_type == "bis":
        try:
            BISCitationOutput.model_validate(output_d)
            results["bis_schema"] = (True, 1.0, "")
        except ValidationError as e:
            results["bis_schema"] = (False, 0.0, str(e.error_count()))
        results["bis_is_lookup"] = verify_bis_is(output_d.get("is_number", ""))
    elif task_type == "tcode":
        try:
            TCodeOutput.model_validate(output_d)
            results["tcode_schema"] = (True, 1.0, "")
        except ValidationError as e:
            results["tcode_schema"] = (False, 0.0, str(e.error_count()))
        results["tcode_resolution"] = verify_tcode(output_d.get("tcode", ""))
    elif task_type == "sap_pm_draft":
        try:
            SAPPMNotificationOutput.model_validate(output_d)
            results["sap_pm_schema"] = (True, 1.0, "")
        except ValidationError as e:
            results["sap_pm_schema"] = (False, 0.0, str(e.error_count()))
        if "tcode" in output_d:
            results["tcode_resolution"] = verify_tcode(output_d["tcode"])

    # Confidence calibration applies to every task type
    answer_correct: bool | None = None
    if "rca_schema_match" in results:
        answer_correct = results["rca_schema_match"][0]
    elif "hsn_top1" in results:
        answer_correct = results["hsn_top1"][0]
    elif "bis_is_lookup" in results:
        answer_correct = results["bis_is_lookup"][0]
    elif "tcode_resolution" in results:
        answer_correct = results["tcode_resolution"][0]
    results["confidence_calibration"] = verify_confidence_calibration(
        output_d, answer_correct=answer_correct
    )

    return results
