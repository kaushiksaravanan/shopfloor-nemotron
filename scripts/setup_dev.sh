#!/usr/bin/env bash
# scripts/setup_dev.sh — one-shot developer environment bootstrap.
#
# Creates a Python venv via `uv`, installs project + dev deps, ensures
# the BIS / HSN / SAP T-code CSVs exist (downloading from public sources
# if a teammate's clone is missing them), then prints the next-step
# commands so a new contributor can run a smoke test inside 5 minutes.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

log() { echo "[setup_dev] $*"; }

# --------------------------------------------------------------------------- #
# 1. uv + venv
# --------------------------------------------------------------------------- #
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv (Astral fast Python installer)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d .venv ]]; then
    log "Creating .venv with uv (Python 3.11+)"
    uv venv .venv --python ">=3.11"
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# --------------------------------------------------------------------------- #
# 2. Install project + optional dev deps
# --------------------------------------------------------------------------- #
log "Installing project dependencies (editable)…"
uv pip install -e . || uv pip install -r <(grep -E '^[a-zA-Z]' pyproject.toml || true) || true

# Extras that are referenced by the training/serving scripts but kept
# optional in pyproject.toml so the cluster install can pin NeMo wheels.
log "Installing dev extras…"
uv pip install \
    typer \
    httpx \
    gradio \
    "peft>=0.12" \
    "datasets>=2.20" \
    "transformers>=4.46" \
    "accelerate>=1.0" \
    "trl>=0.11" || true

# --------------------------------------------------------------------------- #
# 3. Reference data files (BIS / HSN / SAP T-code)
# --------------------------------------------------------------------------- #
mkdir -p data
declare -A FILES=(
    [data/bis_is_master.csv]="https://raw.githubusercontent.com/datameet/india-government-data/master/bis/bis_is_master.csv"
    [data/hsn_seed.csv]="https://raw.githubusercontent.com/datameet/india-government-data/master/gst/hsn_codes.csv"
    [data/sap_pm_tcodes.csv]="https://raw.githubusercontent.com/SAP-samples/cloud-cap-samples/main/data/sap_pm_tcodes.csv"
)
for path in "${!FILES[@]}"; do
    if [[ -s "${path}" ]]; then
        log "OK ${path} (already present, $(wc -l < "${path}") lines)"
        continue
    fi
    log "Downloading ${path}…"
    if ! curl -fsSL "${FILES[$path]}" -o "${path}"; then
        log "WARN: could not download ${path} — keep the stub committed in repo."
        # Write a 1-row stub so downstream scripts don't crash
        case "${path}" in
            *bis*)  echo "is_number,title" > "${path}"; echo "IS 14543,Packaged Natural Mineral Water" >> "${path}";;
            *hsn*)  echo "hsn,description,gst_pct" > "${path}"; echo "84821010,Ball bearings,18" >> "${path}";;
            *tcode*) echo "tcode,description" > "${path}"; echo "IW21,Create maintenance notification" >> "${path}";;
        esac
    fi
done

# --------------------------------------------------------------------------- #
# 4. Pre-create curated/synthetic dirs
# --------------------------------------------------------------------------- #
mkdir -p data/curated data/synthetic runs outputs/sft outputs/grpo engines

# --------------------------------------------------------------------------- #
# 5. Next-step hint
# --------------------------------------------------------------------------- #
cat <<EOF

setup_dev.sh OK.

Next:
  source .venv/bin/activate         # or .venv\\Scripts\\activate on Windows
  bash scripts/repro.sh             # 10-min CPU-friendly end-to-end smoke
  python -m train.sft --dry-run     # 10-step LoRA on tiny stand-in model
  python -m train.rl_grpo --dry-run # 5-step mock-reward GRPO loop
  python -m serve.ghost_demo        # booth UI on http://localhost:7860

EOF
