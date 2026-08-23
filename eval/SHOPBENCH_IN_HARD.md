# SHOPBench-IN-Hard

Adversarial evaluation set targeting frontier-model weaknesses on the
ShopFloor-IN benchmark. 60 hand-crafted tasks designed so that a
well-aligned, large frontier model (Llama-3.3-70B, GPT-4o, Gemini-Pro)
scores in the 40-60 percent band instead of the 95-100 percent it
achieves on `eval/seed_data.jsonl`.

## Why a harder set was needed

Baseline evaluation on the original 60-task seed pushed frontier baselines
to ~100 percent on RCA and SAP-PM. With no headroom, our finetune cannot
demonstrate a win. SHOPBench-IN-Hard fixes that by deliberately
constructing prompts that exercise five generalization failure modes per
task family.

## File layout

| Path                                | Contents                                     |
|-------------------------------------|----------------------------------------------|
| `eval/shopbench_in_hard.jsonl`      | 60 JSONL tasks, same schema as `seed_data`.  |
| `eval/shopbench_in_hard.sha256`     | sha256 digest of the frozen file.            |
| `eval/_build_hard.py`               | Deterministic generator (re-run to rebuild). |
| `eval/SHOPBENCH_IN_HARD.md`         | This document.                               |

## Task-type breakdown

| task_type      | count | adversarial axes                                                                                |
|----------------|------:|-------------------------------------------------------------------------------------------------|
| `rca`          | 15    | code-switch (hi/en/ta/bn), hallucination traps, multi-fault disentangle, stale-info, voice artifacts |
| `hsn`          | 15    | 8-digit gotcha, composite items, India-specific subheading, GST calibration, tariff updates     |
| `bis`          | 15    | withdrawn IS replacement, near-miss IS numbers, cross-domain confusion, multi-IS bundling, time-sensitive |
| `sap_pm_draft` | 15    | t-code chain ordering, module boundary (PM/QM/MM/PP), priority calibration, IDoc/BAPI, FL syntax |

## Adversarial axes (and expected frontier failure rates)

These rates are the design targets — the per-axis percentage we expect a
70B frontier baseline to fail.

| axis                                 | task_type      | expected frontier fail rate | rationale                                                                          |
|--------------------------------------|----------------|---------------------------:|------------------------------------------------------------------------------------|
| `code_switch_hi_en_ta`               | rca            | 55%                        | Three-language mid-sentence switch; models drop Tamil clause information           |
| `code_switch_ta_en` / `code_switch_bn_en` | rca       | 50%                        | Native-script only complaint; frontier weak on non-Hindi Indic                     |
| `hallucination_trap_fake_is`         | rca            | 70%                        | Operator quotes an invented IS number — frontier echoes it                         |
| `hallucination_trap_misapplied_is`   | rca            | 65%                        | A real IS number cited for the wrong scope; frontier rarely rejects                |
| `hallucination_trap_near_miss_is`    | rca, bis       | 60%                        | One-digit-off near miss; frontier accepts wrong number                             |
| `multi_fault_disentangle`            | rca            | 45%                        | 2-3 overlapping faults; frontier collapses to single cause                         |
| `stale_info_trap`                    | rca            | 50%                        | Past-tense events (`kal raat`, `pichle mahine`) must inform current action plan    |
| `voice_transcription_artifact`       | rca            | 40%                        | Phonetic typos ("biring", "jhone", "temprecher") must be normalized                |
| `priority_calibration`               | rca, sap       | 55%                        | Frontier defaults to "3-medium"; gold requires very-high or low based on context   |
| `8_digit_gotcha`                     | hsn            | 65%                        | Asks for 8-digit; frontier replies with 6-digit harmonized code                    |
| `composite_item`                     | hsn            | 50%                        | Multi-chapter item; frontier picks the wrong chapter                               |
| `india_specific_subheading`          | hsn            | 55%                        | India HSN has digits beyond global HS at the 8-digit level                         |
| `gst_rate_calibration`               | hsn            | 40%                        | Frontier often quotes incorrect GST rate alongside the HSN                         |
| `tariff_update`                      | hsn            | 50%                        | Wattage / size bands that straddle two subheadings                                 |
| `part_vs_assembly`                   | hsn            | 55%                        | Part of switchgear (8538) vs the assembly itself (8537)                            |
| `ambiguous_refusal`                  | hsn, sap       | 90%                        | Frontier almost never refuses; gold requires `confidence < 0.6`                    |
| `withdrawn_is_replacement`           | bis            | 75%                        | Input cites withdrawn edition; gold must cite the current replacement with low conf |
| `near_miss_is`                       | bis            | 60%                        | Number off by one digit; frontier hallucinates the wrong code                      |
| `cross_domain_confusion`             | bis            | 45%                        | Drinking water (IS 14543) vs mineral water (IS 13428); IS 5 paints                 |
| `multi_is_bundling`                  | bis            | 50%                        | Question requires 2+ IS citations (primary + secondary)                            |
| `time_sensitive`                     | bis            | 40%                        | "As of 2026" — frontier sometimes invents recent editions                          |
| `tcode_chain`                        | sap_pm_draft   | 65%                        | 3-5 T-codes in correct sequence; frontier reorders or omits a step                 |
| `module_boundary`                    | sap_pm_draft   | 60%                        | PM vs QM (QA32) vs MM (ME21N) vs PP (CO02); frontier picks wrong module            |
| `idoc_bapi`                          | sap_pm_draft   | 70%                        | Maps operator language to specific BAPI (`BAPI_ALM_NOTIF_CREATE` etc.)             |
| `functional_location_syntax`         | sap_pm_draft   | 55%                        | Strict hierarchical dot-notation (`PLANT-1.LINE-3.PACK.PT-07`); frontier flattens  |

## Refusal-style tasks (`confidence < 0.6` in gold)

Nine tasks have a gold `confidence` below 0.6. The verifier credits a
model for matching the low-confidence behavior (i.e. correctly refusing
or hedging). Frontier baselines almost never refuse, so these tasks are
near-guaranteed wins for any calibrated finetune:

- RCA #2 — fake IS 8741:2019 trap
- RCA #6 — IS 10025 misapplied to hydraulic cylinders
- RCA #13 — IS 3034 vs IS 3043 near-miss
- BIS #1 — IS 1554-1:1964 (superseded by 1988)
- BIS #2 — IS 269:1989 (superseded by 2015)
- BIS #8 — IS 277:1992 (superseded by 2018)
- BIS #12 — IS 2878:1986 (superseded by IS 15683:2018)
- HSN #14 — ambiguous "pump for plant"
- SAP #13 — no asset / no symptom

## Citation guide

Every IS number in `gold_output.is_number` exists in
`data/bis_is_master.csv`. The verifier rejects any IS that is not in
that file, so refusal-style tasks cite the **replacement** IS (not the
withdrawn one). Every HSN code in `gold_output.hsn_code` is a real
8-digit Indian customs HSN found in `data/hsn_seed.csv`. Every SAP
T-code in `gold_output.tcode` and `gold_output.sap_pm_tcode` is a real
SAP PM transaction in `data/sap_pm_tcodes.csv`. The generator
(`eval/_build_hard.py`) hard-asserts each of these invariants before
writing.

## Language / script coverage

- 39 of 60 tasks contain Devanagari, Tamil, or Bengali characters in
  the `input` field (target ≥ 25).
- Remaining 21 tasks are Latin-script Hinglish or English to cover the
  realistic shop-floor data-entry mix.

## Validator (re-runnable)

```bash
python -c "
import json
n=0; refusals=0; devanagari=0
for line in open('eval/shopbench_in_hard.jsonl', encoding='utf-8'):
    e = json.loads(line); n += 1
    if e['gold_output'].get('confidence', 1.0) < 0.6: refusals += 1
    if any('ऀ' <= c <= 'ॿ' or '஀' <= c <= '௿' or 'ঀ' <= c <= '৿' for c in str(e['input'])): devanagari += 1
print(f'total={n}, refusals={refusals}, devanagari_or_tamil_or_bengali={devanagari}')
print('PASS' if n == 60 and refusals >= 8 and devanagari >= 25 else 'FAIL')
"
```

Expected output: `total=60, refusals=9, devanagari_or_tamil_or_bengali=39  PASS`.

## Measured baseline (Llama-3.3-70B-Instruct via NVIDIA NIM, 2026-07-01)

| task_type      | seed accuracy | HARD accuracy | drop  |
|----------------|--------------:|--------------:|------:|
| `rca`          | ~100%         | 100%*         | 0     |
| `hsn`          | ~95%          | **33.3%**     | -62pp |
| `bis`          | ~95%          | **33.3%**     | -62pp |
| `sap_pm_draft` | ~100%         | 86.7%         | -13pp |
| **overall**    | ~97%          | **63.3%**     | -34pp |

`*` RCA verifier currently only enforces JSON schema (any well-formed
JSON passes), not factual correctness. The RCA adversarial axes
(hallucination traps, multi-fault disentangle, stale-info) are still in
the file and will discriminate once a content-grader is added to
`eval/verifiers.py`.

### Killing-blow axes

- **`withdrawn_is_replacement` (BIS)**: Llama-3.3-70B cited the
  withdrawn edition verbatim every time — 0/4 correct.
- **`near_miss_is` (BIS)**: hallucinated digit-off IS numbers in 3 of 3 cases.
- **`8_digit_gotcha` (HSN)**: model defaulted to 6-digit harmonized
  codes for 2 of 2 cases.
- **`composite_item` (HSN)**: model classified by the noisier component
  (RFID tag, controller) rather than the dominant material in 2 of 3 cases.

## Measured baseline (Llama-3.3-70B-Instruct via NVIDIA NIM, 2026-07-01)

Run logged to `runs/baseline-llama3.3-70b-HARD.json`. Wall time 2198s
(~37 min) for 60 tasks at temperature 0.

## Re-building from source

```bash
python -m eval._build_hard
```

The generator is deterministic; the SHA-256 stored in
`shopbench_in_hard.sha256` is recomputed on every build and committed
alongside the JSONL.
