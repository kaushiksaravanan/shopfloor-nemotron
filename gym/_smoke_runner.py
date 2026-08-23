"""E2E smoke runner — hits the running NeMo Gym server and dumps a full transcript."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8765"
LOG = Path(__file__).resolve().parent / "e2e_smoke.log"


def w(fh, label: str, *parts: str) -> None:
    fh.write(f"\n========== {label} ==========\n")
    for p in parts:
        fh.write(p.rstrip() + "\n")


def show(resp: httpx.Response) -> str:
    try:
        body = json.dumps(resp.json(), indent=2, ensure_ascii=False)
    except Exception:
        body = resp.text
    return f"{resp.request.method} {resp.request.url}\n  -> HTTP {resp.status_code}\n{body}"


def main() -> int:
    fh = LOG.open("w", encoding="utf-8")
    fh.write(f"# NeMo Gym E2E smoke transcript — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.write(f"# Target: {BASE}\n")

    client = httpx.Client(base_url=BASE, timeout=10.0)
    fails: list[str] = []

    # ------------------------------------------------------------------- a
    r = client.get("/healthz")
    w(fh, "a) GET /healthz", show(r))
    if r.status_code != 200 or r.json().get("status") != "ok":
        fails.append("a")

    # ------------------------------------------------------------------- b
    r = client.post("/seed_session", json={"task_type": "rca"})
    w(fh, "b) POST /seed_session (rca)", "request body: " + json.dumps({"task_type": "rca"}), show(r))
    if r.status_code != 200:
        fails.append("b")
        print("seed failed; aborting"); fh.close(); return 1
    seed_body = r.json()
    sid_rca = seed_body["session_id"]
    fh.write(f"\nCaptured session_id = {sid_rca}\n")
    fh.write(f"task_type = {seed_body['initial_state']['task_type']}\n")
    fh.write(f"prompt    = {seed_body['initial_state']['prompt']!r}\n")

    # ------------------------------------------------------------------- c
    body_c = {"code": "84821010"}
    r = client.post("/my_tool/hsn_lookup", json=body_c)
    w(fh, "c) POST /my_tool/hsn_lookup (84821010 = ball bearings)",
      "request body: " + json.dumps(body_c), show(r))
    if r.status_code != 200 or not r.json().get("ok"):
        fails.append("c")

    # ------------------------------------------------------------------- d
    body_d = {"code": "IS 14543:2004"}
    r = client.post("/my_tool/bis_is_lookup", json=body_d)
    w(fh, "d) POST /my_tool/bis_is_lookup (IS 14543:2004)",
      "request body: " + json.dumps(body_d), show(r))
    if r.status_code != 200 or not r.json().get("ok"):
        fails.append("d")

    # ------------------------------------------------------------------- e
    # Use a fresh session for each /verify (one-shot env terminates the session).
    seed = client.post("/seed_session", json={"task_type": "rca"}).json()
    sid_e = seed["session_id"]
    gold_rca = {
        "asset_id": "PA-204",
        "symptom": "Compressor bearing seized with high vibration and abnormal noise",
        "root_cause": "Bearing failure due to lubrication starvation or contamination",
        "corrective_action": "Stop machine, lockout-tagout, replace bearing assembly, flush lubrication circuit, verify alignment before restart",
        "severity": "high",
        "confidence": 0.88,
        "sap_pm_tcode": "IW21",
    }
    body_e = {"session_id": sid_e, "model_output": json.dumps(gold_rca)}
    r = client.post("/verify", json=body_e)
    w(fh, "e) POST /verify (gold RCA output from seed_data.jsonl line 1)",
      "session_id = " + sid_e,
      "model_output (parsed) = " + json.dumps(gold_rca, indent=2),
      show(r))
    if r.status_code != 200 or r.json().get("reward", 0) <= 0.7:
        fails.append(f"e (reward={r.json().get('reward')})")

    # ------------------------------------------------------------------- f
    seed = client.post("/seed_session", json={"task_type": "rca"}).json()
    sid_f = seed["session_id"]
    body_f = {"session_id": sid_f, "model_output": {}}
    r = client.post("/verify", json=body_f)
    w(fh, "f) POST /verify (empty dict — anti-reward-hacking)",
      "request body: " + json.dumps(body_f), show(r))
    if r.status_code != 200 or r.json().get("reward") != 0.0:
        fails.append(f"f (reward={r.json().get('reward')})")

    # ------------------------------------------------------------------- g
    # Schema-match-only: valid BIS shape but bogus IS number + over-confident.
    seed = client.post("/seed_session", json={"task_type": "bis"}).json()
    sid_g = seed["session_id"]
    schema_only = {
        "is_number": "IS 99999:2099",   # not in fallback table -> verify_bis_is = 0
        "title": "Fictional Standard",
        "domain": "manufacturing",
        "confidence": 0.95,             # over-confident on wrong answer -> conf_cal = 0
    }
    body_g = {"session_id": sid_g, "model_output": json.dumps(schema_only)}
    r = client.post("/verify", json=body_g)
    w(fh, "g) POST /verify (schema-only, garbage values — reward must be <= 0.30)",
      "session_id = " + sid_g,
      "model_output (parsed) = " + json.dumps(schema_only, indent=2),
      show(r))
    if r.status_code != 200 or r.json().get("reward", 999) > 0.3 + 1e-9:
        fails.append(f"g (reward={r.json().get('reward')})")

    # --------------------------------------------------------------------
    fh.write("\n\n========== SUMMARY ==========\n")
    if fails:
        fh.write(f"FAILURES ({len(fails)}): {fails}\n")
    else:
        fh.write("All 7 endpoint cases passed.\n")
    fh.close()
    print("FAILURES:" if fails else "ALL PASS", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
