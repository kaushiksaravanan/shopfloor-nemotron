"""
tests/test_gym_server.py
========================
Smoke + correctness tests for the NeMo Gym FastAPI server.

Covers:
  - /healthz returns ok
  - /seed_session returns a valid task + session id
  - /my_tool/hsn_lookup returns a sane response
  - /my_tool/bis_is_lookup against a known IS in the master CSV
  - /my_tool/tcode_lookup against IW21
  - /verify computes a positive reward for a well-formed RCA output
  - /verify rejects empty / null / "{}" outputs (anti-reward-hacking)
  - /verify caps schema-only outputs at SCHEMA_ONLY_CAP (anti-reward-hacking)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path so `gym.server` and `eval.*` import cleanly
# whether pytest is invoked from repo root or from /tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gym.server import SCHEMA_ONLY_CAP, app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------- #
# /healthz
# --------------------------------------------------------------------------- #

def test_healthz_returns_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "sessions" in body


# --------------------------------------------------------------------------- #
# /seed_session
# --------------------------------------------------------------------------- #

def test_seed_session_returns_valid_task(client: TestClient) -> None:
    r = client.post("/seed_session", json={"task_type": "rca"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "session_id" in body
    assert body["initial_state"]["task_type"] == "rca"
    assert body["initial_state"]["prompt"]
    assert body["initial_state"]["expected_output_schema"]["required"]


def test_seed_session_unknown_task_404s(client: TestClient) -> None:
    r = client.post("/seed_session", json={"task_type": "rca"})
    assert r.status_code == 200
    # task_type validation comes from the Pydantic Literal; bogus type → 422
    r2 = client.post("/seed_session", json={"task_type": "not_a_real_type"})
    assert r2.status_code == 422


# --------------------------------------------------------------------------- #
# /my_tool/*
# --------------------------------------------------------------------------- #

def test_my_tool_hsn_lookup_valid(client: TestClient) -> None:
    r = client.post("/my_tool/hsn_lookup", json={"code": "72162100"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["hsn_code"] == "72162100"
    assert body["result"]["chapter"] == "72"


def test_my_tool_hsn_lookup_rejects_garbage(client: TestClient) -> None:
    r = client.post("/my_tool/hsn_lookup", json={"code": "abc"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_my_tool_bis_is_lookup_known(client: TestClient) -> None:
    # IS 2062:2011 is in the fallback table inside eval/verifiers.py.
    r = client.post("/my_tool/bis_is_lookup", json={"code": "IS 2062:2011"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["reward"] == 1.0


def test_my_tool_tcode_lookup_iw21(client: TestClient) -> None:
    r = client.post("/my_tool/tcode_lookup", json={"code": "IW21"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_my_tool_unknown_404s(client: TestClient) -> None:
    r = client.post("/my_tool/does_not_exist", json={})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# /verify — positive case for RCA
# --------------------------------------------------------------------------- #

_GOOD_RCA = {
    # Must match eval.shopbench_in_spec.RCAOutput exactly (extra="forbid").
    "asset_id": "CNC-LINE3-01",
    "symptom": "Spindle trips intermittently mid-shift.",
    "root_cause": "Worn spindle bearing causing thermal overload.",
    "corrective_action": "1) LOTO. 2) Replace bearing (SKF 7008). 3) Re-balance.",
    "severity": "high",
    "confidence": 0.82,
    "sap_pm_tcode": "IW21",
}


def test_verify_rewards_good_rca_output(client: TestClient) -> None:
    seed = client.post("/seed_session", json={"task_type": "rca"}).json()
    sid = seed["session_id"]

    r = client.post("/verify", json={
        "session_id": sid,
        "model_output": json.dumps(_GOOD_RCA),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["terminated"] is True
    # Should clear the schema-only cap and earn non-trivial reward.
    assert body["reward"] > SCHEMA_ONLY_CAP, body
    assert body["info"]["per_component"]["schema_match"] > 0.0
    assert body["info"]["per_component"]["conf_cal"] >= 0.0


# --------------------------------------------------------------------------- #
# /verify — anti-reward-hacking cases
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("hacked", ["", "{}", "null", "[]", "  ", '{"x": 1}'])
def test_verify_empty_or_near_empty_json_returns_zero(client: TestClient, hacked: str) -> None:
    seed = client.post("/seed_session", json={"task_type": "rca"}).json()
    sid = seed["session_id"]
    r = client.post("/verify", json={"session_id": sid, "model_output": hacked})
    assert r.status_code == 200
    body = r.json()
    assert body["reward"] == 0.0, f"reward leak on hacked output {hacked!r}: {body}"


def test_verify_unknown_session_404s(client: TestClient) -> None:
    r = client.post("/verify", json={
        "session_id": "00000000-0000-0000-0000-000000000000",
        "model_output": "{}",
    })
    assert r.status_code == 404


def test_verify_schema_only_capped(client: TestClient) -> None:
    """A well-shaped JSON that passes the Pydantic schema but has bogus content
    (mis-calibrated confidence, no real lookup hit) must be capped."""
    seed = client.post("/seed_session", json={"task_type": "bis"}).json()
    sid = seed["session_id"]
    # BISCitationOutput requires is_number starting with 'IS '. We provide a
    # plausibly-shaped but unknown IS — verify_bis_is returns 0.0, so the only
    # passing verifier is the schema. Reward must be <= SCHEMA_ONLY_CAP.
    payload = {
        "is_number": "IS 99999:2099",
        "title": "Fictional Standard",
        "domain": "manufacturing",
        # over-confident on a wrong answer → conf_cal = 0.0
        "confidence": 0.95,
    }
    r = client.post("/verify", json={
        "session_id": sid,
        "model_output": json.dumps(payload),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["reward"] <= SCHEMA_ONLY_CAP + 1e-9, body
