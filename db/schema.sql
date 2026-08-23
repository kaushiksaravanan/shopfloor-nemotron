-- ShopFloor-Nemotron results ledger schema (SQLite).
-- Apache-2.0. Single source of truth for every experiment.

CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  ts            INTEGER NOT NULL,         -- unix epoch seconds
  kind          TEXT NOT NULL,            -- 'eval' | 'sft' | 'grpo' | 'quant' | 'data-gen' | 'baseline'
  model         TEXT NOT NULL,
  base_model    TEXT,
  provider      TEXT,                     -- 'groq', 'gemini', 'local', 'nim', etc.
  dataset_version TEXT,
  dataset_sha256 TEXT,
  n_examples    INTEGER,
  params_json   TEXT NOT NULL,            -- full hyperparam JSON
  metrics_json  TEXT NOT NULL,            -- {"overall": 0.667, "rca": 1.0, ...}
  artifact_path TEXT,
  git_commit    TEXT,
  host          TEXT,
  elapsed_s     REAL,
  status        TEXT DEFAULT 'completed', -- 'completed' | 'failed' | 'aborted'
  notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_kind_model ON runs(kind, model);
CREATE INDEX IF NOT EXISTS idx_ts ON runs(ts DESC);

CREATE VIEW IF NOT EXISTS leaderboard_eval AS
  SELECT
    run_id,
    datetime(ts, 'unixepoch', 'localtime') AS ts_local,
    model, provider, dataset_version, n_examples,
    json_extract(metrics_json, '$.overall')  AS overall,
    json_extract(metrics_json, '$.rca')      AS rca,
    json_extract(metrics_json, '$.hsn')      AS hsn,
    json_extract(metrics_json, '$.bis')      AS bis,
    json_extract(metrics_json, '$.sap_pm')   AS sap_pm,
    artifact_path
  FROM runs
  WHERE kind IN ('eval', 'baseline')
  ORDER BY overall DESC NULLS LAST, ts DESC;

CREATE VIEW IF NOT EXISTS leaderboard_sft AS
  SELECT
    run_id,
    datetime(ts, 'unixepoch', 'localtime') AS ts_local,
    model, base_model, n_examples,
    json_extract(metrics_json, '$.final_loss')    AS final_loss,
    json_extract(metrics_json, '$.tokens_per_s')  AS tok_per_s,
    json_extract(params_json,  '$.lora_rank')     AS lora_rank,
    json_extract(params_json,  '$.epochs')        AS epochs,
    artifact_path
  FROM runs
  WHERE kind = 'sft'
  ORDER BY final_loss ASC NULLS LAST, ts DESC;
