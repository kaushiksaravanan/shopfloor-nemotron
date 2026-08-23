# ShopFloor-Nemotron — Makefile
# Apache-2.0. Track B — India Agentic AI Open Hackathon 2026.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY      ?= python
UV      ?= uv
WANDB   ?= wandb
OUT     ?= outputs
DATA    ?= data
HF_REPO ?= sap-labs-india/shopfloor-nemotron

.PHONY: help setup lint test gen-data curate train-sft train-rl eval quant deploy-jetson deploy-nim repro all clean db-init db-ingest leaderboard refresh-numbers

help:  ## Print this help and list every target.
	@awk 'BEGIN {FS = ":.*##"; printf "\nShopFloor-Nemotron targets:\n\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

setup:  ## Install Python deps + NeMo toolkit against the local CUDA wheel.
	$(UV) sync --extra train --extra serve --extra dev && bash scripts/install_nemo.sh

lint:  ## Run ruff check + format.
	$(UV) run ruff check . && $(UV) run ruff format --check .

test:  ## Run pytest with coverage.
	$(UV) run pytest --cov=src/shopfloor_nemotron --cov-report=term-missing

gen-data:  ## Synthesise 18k Hinglish/Tamil complaint→ticket pairs with NeMo Data Designer.
	$(UV) run $(PY) -m shopfloor_nemotron.data.designer --seeds $(DATA)/seeds.jsonl --n 18000 --out $(DATA)/synthetic/raw.jsonl

curate:  ## Deduplicate + quality-filter the synthetic set with NeMo Curator.
	$(UV) run $(PY) -m shopfloor_nemotron.data.curator --in $(DATA)/synthetic/raw.jsonl --out $(DATA)/curated/sft.parquet

train-sft:  ## LoRA SFT Nemotron 3 Nano 9B with NeMo AutoModel.
	$(UV) run $(PY) -m shopfloor_nemotron.train.sft --config train/configs/sft.yaml --out $(OUT)/sft

train-rl:  ## GRPO against the FastAPI NeMo Gym env.
	$(UV) run $(PY) -m shopfloor_nemotron.train.rl --config train/configs/grpo.yaml --sft-ckpt $(OUT)/sft --out $(OUT)/rl

eval:  ## Score the current checkpoint on SHOPBench-IN.
	$(UV) run $(PY) -m shopfloor_nemotron.eval.shopbench --ckpt $(OUT)/rl --split dev --report $(OUT)/eval/report.json

quant:  ## NVFP4 quantize + TensorRT-LLM compile.
	$(UV) run $(PY) -m shopfloor_nemotron.quant.nvfp4 --ckpt $(OUT)/rl --engine $(OUT)/engine

deploy-jetson:  ## Push engine to Jetson Orin Nano over SSH and run smoke test.
	bash scripts/deploy_jetson.sh $(OUT)/engine

deploy-nim:  ## Deploy NIM container to A100 as cloud fallback.
	bash scripts/deploy_nim.sh $(OUT)/rl

repro:  ## End-to-end: setup → gen-data → curate → SFT → RL → eval → quant.
	$(MAKE) setup && $(MAKE) gen-data && $(MAKE) curate && $(MAKE) train-sft && $(MAKE) train-rl && $(MAKE) eval && $(MAKE) quant

all: lint test repro  ## Lint, test, and full repro.

clean:  ## Remove caches and outputs (keeps data/raw and data/curated).
	rm -rf $(OUT) wandb .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

db-init:  ## Ingest runs/*.json + outputs/sft/**/metrics.json into db/results.sqlite.
	@$(PY) -m db.leaderboard ingest

db-ingest: db-init  ## Alias for db-init.

leaderboard:  ## Print the current results leaderboard.
	@$(PY) -m db.leaderboard

refresh-numbers:  ## Re-render README + deck slide 10 from the DB.
	@$(PY) -m db.refresh_artifacts
