# Licensed under the Apache License, Version 2.0
"""Unit tests for the verifier suite. CPU-only, no GPU needed."""
from __future__ import annotations

import json

import pytest

from eval.verifiers import (
    verify_bis_is,
    verify_confidence_calibration,
    verify_hsn_top1,
    verify_json_parseable,
    verify_rca_schema,
    verify_tcode,
    verify_task,
)


# ---------------------------------------------------------------------------
# RCA schema
# ---------------------------------------------------------------------------

_VALID_RCA = {
    "asset_id": "PA-204",
    "symptom": "Compressor bearing seized",
    "root_cause": "Lubrication failure",
    "corrective_action": "Replace bearing assembly",
    "severity": "high",
    "confidence": 0.88,
    "sap_pm_tcode": "IW21",
}


def test_valid_rca_json_passes() -> None:
    ok, reward, err = verify_rca_schema(_VALID_RCA)
    assert ok is True
    assert reward == 1.0
    assert err == ""


def test_valid_rca_json_string_form_passes() -> None:
    ok, reward, _ = verify_rca_schema(json.dumps(_VALID_RCA))
    assert ok is True
    assert reward == 1.0


@pytest.mark.parametrize(
    "missing_field",
    ["asset_id", "symptom", "root_cause", "corrective_action", "severity", "confidence"],
)
def test_missing_required_field_fails(missing_field: str) -> None:
    bad = {k: v for k, v in _VALID_RCA.items() if k != missing_field}
    ok, reward, err = verify_rca_schema(bad)
    assert ok is False
    assert reward < 1.0
    assert "schema_error" in err


def test_unparseable_json_fails() -> None:
    ok, _, err = verify_json_parseable("{not valid json")
    assert ok is False
    assert "json_parse_error" in err


def test_bad_severity_fails() -> None:
    bad = {**_VALID_RCA, "severity": "extremely-high"}
    ok, _, _ = verify_rca_schema(bad)
    assert ok is False


def test_confidence_out_of_range_fails() -> None:
    bad = {**_VALID_RCA, "confidence": 1.5}
    ok, _, _ = verify_rca_schema(bad)
    assert ok is False


# ---------------------------------------------------------------------------
# BIS lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "is_number",
    [
        "IS 14543:2004",
        "IS 11196:1985",
        "IS 9000:2006",
        "IS 16046:2018",
        "IS 4985:2000",
    ],
)
def test_valid_is_number_passes(is_number: str) -> None:
    ok, reward, _ = verify_bis_is(is_number)
    assert ok is True
    assert reward >= 0.9


@pytest.mark.parametrize(
    "is_number",
    ["IS 99999:9999", "IS 0:0000", "not a number", ""],
)
def test_unknown_is_rejected(is_number: str) -> None:
    ok, reward, err = verify_bis_is(is_number)
    assert ok is False
    assert reward == 0.0
    assert "unknown" in err or "must be str" in err


def test_is_number_with_extra_spaces_normalizes() -> None:
    ok, reward, _ = verify_bis_is("IS  14543:2004")
    assert ok is True
    assert reward == 0.9


# ---------------------------------------------------------------------------
# HSN top-1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "predicted,gold,expected_reward",
    [
        ("84821010", "84821010", 1.0),  # exact
        ("84821020", "84821010", 0.5),  # same subheading (84821x)
        ("84822000", "84821010", 0.2),  # same heading (8482xx)
        ("85013120", "84821010", 0.0),  # different chapter
    ],
)
def test_hsn_top1_grading(predicted: str, gold: str, expected_reward: float) -> None:
    _, reward, _ = verify_hsn_top1(predicted, gold)
    assert reward == expected_reward


def test_correct_hsn_top1_returns_full_reward() -> None:
    ok, reward, err = verify_hsn_top1("84137010", "84137010")
    assert ok is True
    assert reward == 1.0
    assert err == ""


def test_wrong_hsn_top1_returns_zero_reward() -> None:
    ok, reward, _ = verify_hsn_top1("99999999", "84137010")
    assert ok is False
    assert reward == 0.0


def test_hsn_invalid_format_fails() -> None:
    ok, _, err = verify_hsn_top1("ABC12345", "84821010")
    assert ok is False
    assert "8-digit" in err


# ---------------------------------------------------------------------------
# T-code resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tcode", ["IW21", "IW31", "IK11", "IH01"])
def test_real_tcode_passes(tcode: str) -> None:
    ok, reward, _ = verify_tcode(tcode)
    assert ok is True
    assert reward == 1.0


def test_unknown_tcode_rejected() -> None:
    ok, reward, err = verify_tcode("ZZ99")
    assert ok is False
    assert reward == 0.0
    assert "unknown" in err


def test_tcode_lowercase_normalizes() -> None:
    ok, _, _ = verify_tcode("iw21")
    assert ok is True


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def test_high_conf_correct_is_calibrated() -> None:
    out = {"confidence": 0.9}
    ok, reward, _ = verify_confidence_calibration(out, answer_correct=True)
    assert ok is True
    assert reward == 1.0


def test_low_conf_wrong_is_calibrated_refusal() -> None:
    """Spec line: confidence < 0.6 with a wrong answer counts as a proper refuse."""
    out = {"confidence": 0.4}
    ok, reward, msg = verify_confidence_calibration(out, answer_correct=False)
    assert ok is True
    assert reward == 1.0
    assert "refused" in msg


def test_high_conf_wrong_is_miscalibrated() -> None:
    out = {"confidence": 0.95}
    ok, reward, _ = verify_confidence_calibration(out, answer_correct=False)
    assert ok is False
    assert reward == 0.0


def test_low_conf_correct_is_under_confident() -> None:
    out = {"confidence": 0.3}
    ok, reward, _ = verify_confidence_calibration(out, answer_correct=True)
    assert ok is False
    assert 0.0 < reward < 1.0


def test_missing_confidence_fails() -> None:
    ok, _, err = verify_confidence_calibration({}, answer_correct=True)
    assert ok is False
    assert "missing" in err


# ---------------------------------------------------------------------------
# Routed verify_task
# ---------------------------------------------------------------------------

def test_verify_task_rca_end_to_end() -> None:
    results = verify_task("rca", _VALID_RCA, gold={})
    assert results["rca_schema_match"][0] is True
    assert "confidence_calibration" in results


def test_verify_task_hsn_exact_match() -> None:
    out = {"hsn_code": "84821010", "description": "Ball Bearings", "gst_rate": 18.0, "confidence": 0.95}
    results = verify_task("hsn", out, gold={"hsn_code": "84821010"})
    assert results["hsn_top1"][0] is True
    assert results["hsn_top1"][1] == 1.0


def test_verify_task_bis_lookup_success() -> None:
    out = {"is_number": "IS 14543:2004", "title": "x", "domain": "food", "confidence": 0.9}
    results = verify_task("bis", out, gold={})
    assert results["bis_is_lookup"][0] is True
