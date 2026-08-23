"""LoRA SFT trainer for ShopFloor-Nemotron.

Why NeMo AutoModel over Megatron-Core?
--------------------------------------
Per the NVIDIA workshop deck (and confirmed by NeMo 2.0 docs):

  - <1000 GPU rule: AutoModel is the recommended path for any fine-tune
    that fits within a thousand GPUs. Megatron-Core only pays off past
    that scale (its tensor/sequence parallel rewrite cost is amortised
    over very large pre-training runs).
  - ~2.5x faster than vanilla HF Trainer on H100 because AutoModel
    uses fused FSDP2 + flash-attn-3 + selective activation checkpointing
    out of the box.
  - ~1.2x faster than Unsloth on the same hardware, with multi-node
    support (Unsloth is single-GPU only).
  - Same HuggingFace-shaped checkpoint in and out -> trivial to plug
    into TensorRT-Model-Optimizer for the NVFP4 quant step downstream.

This script tries NeMo AutoModel first. If `nemo_toolkit` is not
installed (laptop / CI / teammate without GPU), it transparently falls
back to `transformers` + `peft` so the *logic* can still be iterated on.
The actual training run on the cluster will use NeMo.

Usage
-----
    # full run on cluster
    python -m train.sft --model nvidia/Nemotron-Nano-9B-v2 \
        --train-file data/curated/train.jsonl --output-dir outputs/sft

    # 10-step laptop smoke test on synthetic data
    python -m train.sft --dry-run
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.sft")

# --------------------------------------------------------------------------- #
# Backend detection: NeMo AutoModel preferred, HF+PEFT fallback
# --------------------------------------------------------------------------- #
_NEMO_AVAILABLE = False
try:
    # NeMo 2.0 AutoModel entry point
    import nemo_toolkit  # noqa: F401  (presence check)
    from nemo.collections import llm as _nemo_llm  # type: ignore
    _NEMO_AVAILABLE = True
    log.info("NeMo AutoModel backend detected (preferred path).")
except Exception:  # pragma: no cover - environment dependent
    log.warning(
        "NeMo not installed; using HF+PEFT fallback. "
        "NeMo AutoModel will be used on the actual training cluster. "
        "To install: `pip install nemo_toolkit[all]`."
    )

# HF stack is always present (declared in pyproject.toml dependencies)
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

app = typer.Typer(add_completion=False, help="ShopFloor-Nemotron LoRA SFT")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SFTConfig:
    model: str
    train_file: Path
    output_dir: Path
    rank: int
    alpha: int
    dropout: float
    epochs: int
    learning_rate: float
    warmup_ratio: float
    grad_accum: int
    micro_batch: int
    max_seq_length: int
    seed: int
    dry_run: bool
    push_to_hub: bool
    hub_repo: str | None
    wandb_project: str
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty training file: {path}")
    return rows


def _synthetic_examples(n: int = 64) -> list[dict[str, Any]]:
    """Tiny code-mixed Hinglish samples for --dry-run on a laptop."""
    base = [
        {
            "prompt": "बेयरिंग जाम, P3 line down",
            "response": json.dumps({
                "rca": "bearing seizure due to lubrication failure",
                "bis": "IS 14543",
                "hsn": "84821010",
                "tcode": "IW21",
                "confidence": 0.88,
            }, ensure_ascii=False),
        },
        {
            "prompt": "induction motor 5 kW classify HSN",
            "response": json.dumps({
                "hsn": "85013120", "gst": 18, "confidence": 0.93,
            }),
        },
    ]
    out = []
    for i in range(n):
        ex = dict(base[i % len(base)])
        ex["prompt"] = f"[{i}] " + ex["prompt"]
        out.append(ex)
    return out


def _format_example(ex: dict[str, Any]) -> str:
    return (
        "<|system|>You are ShopFloor-Nemotron, a shop-floor RCA copilot.\n"
        f"<|user|>{ex['prompt']}\n"
        f"<|assistant|>{ex['response']}"
    )


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _train_nemo(cfg: SFTConfig, dataset: Dataset) -> dict[str, float]:
    """NeMo AutoModel path. Exercised on the GPU cluster.

    We deliberately keep this thin: NeMo's recipe API does the heavy
    lifting (FSDP2 device_mesh, fused optims, mixed-precision policy).
    """
    log.info("Running NeMo AutoModel LoRA SFT (rank=%d, alpha=%d).", cfg.rank, cfg.alpha)
    # NB: import inside the function so the HF-only fallback path stays clean.
    from nemo.collections import llm  # type: ignore
    from nemo.collections.llm.peft import LoRA  # type: ignore

    peft_cfg = LoRA(
        target_modules=list(cfg.target_modules),
        dim=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
    )
    recipe = llm.recipes.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=cfg.model,
        dir=str(cfg.output_dir),
        name="shopfloor-sft",
        num_nodes=1,
        peft_scheme=peft_cfg,
    )
    recipe.trainer.max_steps = -1
    recipe.trainer.max_epochs = cfg.epochs
    recipe.optim.config.lr = cfg.learning_rate
    recipe.data.global_batch_size = cfg.grad_accum * cfg.micro_batch
    recipe.data.micro_batch_size = cfg.micro_batch
    t0 = time.time()
    llm.finetune(recipe)
    elapsed = time.time() - t0
    return {"elapsed_s": elapsed, "backend": "nemo"}


def _train_hf(cfg: SFTConfig, dataset: Dataset) -> dict[str, float]:
    """HF + PEFT fallback. Runs on a laptop CPU when --dry-run is set."""
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    log.info("Loading tokenizer + model (%s) via HF fallback.", cfg.model)
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=dtype, trust_remote_code=True,
    )

    # Only attach LoRA to modules that actually exist in this model
    # (target_modules list is Llama-family naming; Nemotron uses the same names).
    present = set()
    for name, _ in model.named_modules():
        for t in cfg.target_modules:
            if name.endswith(t):
                present.add(t)
    if not present:
        log.warning("None of %s found in model; PEFT will pick defaults.", cfg.target_modules)
        present = set(cfg.target_modules)

    peft_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(present),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    def _tok(ex):
        text = _format_example(ex)
        out = tok(
            text, truncation=True, max_length=cfg.max_seq_length, padding=False,
        )
        out["labels"] = list(out["input_ids"])
        return out

    tokenized = dataset.map(_tok, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        save_steps=200,
        save_total_limit=3,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        seed=cfg.seed,
        max_steps=10 if cfg.dry_run else -1,
        gradient_checkpointing=True,
        optim="adamw_torch",
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_repo,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized, data_collator=collator,
    )

    t0 = time.time()
    res = trainer.train()
    elapsed = time.time() - t0

    # Effective tokens/sec
    n_tokens = sum(len(x["input_ids"]) for x in tokenized)
    tok_per_s = n_tokens * cfg.epochs / max(elapsed, 1e-6)
    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    log.info(
        "HF fallback done: %.1fs, %.0f tok/s, peak mem %.2f GiB",
        elapsed, tok_per_s, peak_mem_gb,
    )

    trainer.save_model(str(cfg.output_dir / "final"))
    return {
        "elapsed_s": elapsed,
        "tokens_per_s": tok_per_s,
        "peak_mem_gb": peak_mem_gb,
        "train_loss": float(res.training_loss),
        "backend": "hf",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@app.command()
def main(
    model: str = typer.Option("nvidia/Nemotron-Nano-9B-v2", help="HF model id."),
    train_file: Path = typer.Option(Path("data/curated/train.jsonl")),
    output_dir: Path = typer.Option(Path("outputs/sft")),
    rank: int = typer.Option(32, help="LoRA rank."),
    alpha: int = typer.Option(64),
    dropout: float = typer.Option(0.05),
    epochs: int = typer.Option(3),
    learning_rate: float = typer.Option(2e-4),
    warmup_ratio: float = typer.Option(0.03),
    grad_accum: int = typer.Option(64),
    micro_batch: int = typer.Option(1),
    max_seq_length: int = typer.Option(4096),
    seed: int = typer.Option(42),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="10 steps on synthetic tiny data."
    ),
    push_to_hub: bool = typer.Option(False, "--push-to-hub"),
    hub_repo: str = typer.Option(None, help="HF Hub repo id when --push-to-hub."),
    wandb_project: str = typer.Option("shopfloor-nemotron"),
) -> None:
    cfg = SFTConfig(
        model=model,
        train_file=train_file,
        output_dir=output_dir,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        grad_accum=grad_accum,
        micro_batch=micro_batch,
        max_seq_length=max_seq_length,
        seed=seed,
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
        wandb_project=wandb_project,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    if cfg.dry_run:
        # On --dry-run we use a tiny synthetic dataset *and* a tiny stand-in
        # model so the smoke test finishes in <2 minutes on a laptop CPU.
        if cfg.model == "nvidia/Nemotron-Nano-9B-v2":
            log.warning(
                "--dry-run: substituting tiny stand-in model "
                "(sshleifer/tiny-gpt2) for the 9B Nemotron."
            )
            cfg.model = "sshleifer/tiny-gpt2"
        rows = _synthetic_examples(64)
        log.info("Using %d synthetic examples for dry run.", len(rows))
    else:
        rows = _load_jsonl(cfg.train_file)
        log.info("Loaded %d training rows from %s", len(rows), cfg.train_file)

    dataset = Dataset.from_list(rows)

    # Configure wandb
    if os.getenv("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        log.info("W&B logging enabled: project=%s", cfg.wandb_project)

    if _NEMO_AVAILABLE and not cfg.dry_run:
        metrics = _train_nemo(cfg, dataset)
    else:
        if _NEMO_AVAILABLE and cfg.dry_run:
            log.info("--dry-run: forcing HF backend even though NeMo is present.")
        metrics = _train_hf(cfg, dataset)

    log.info("Training complete. Metrics: %s", metrics)
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Persist to the SQLite results ledger.
    try:
        from db.results import log_run
        run_id = log_run(
            kind="sft",
            model=cfg.model,
            base_model=cfg.model,
            params={
                "lora_rank": cfg.rank,
                "lora_alpha": cfg.alpha,
                "lora_dropout": cfg.dropout,
                "target_modules": list(cfg.target_modules),
                "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate,
                "warmup_ratio": cfg.warmup_ratio,
                "grad_accum": cfg.grad_accum,
                "micro_batch": cfg.micro_batch,
                "max_seq_length": cfg.max_seq_length,
                "seed": cfg.seed,
                "dry_run": cfg.dry_run,
                "backend": metrics.get("backend"),
            },
            metrics={
                "final_loss": metrics.get("train_loss"),
                "tokens_per_s": metrics.get("tokens_per_s"),
                "peak_mem_gb": metrics.get("peak_mem_gb"),
                "elapsed_s": metrics.get("elapsed_s"),
            },
            n_examples=len(rows),
            artifact_path=str(cfg.output_dir),
            elapsed_s=metrics.get("elapsed_s"),
        )
        log.info("Logged run to DB: %s", run_id)
    except Exception as _e:  # noqa: BLE001
        log.warning("DB log skipped: %s", _e)


if __name__ == "__main__":
    try:
        app()
    except Exception as exc:
        log.exception("SFT failed: %s", exc)
        sys.exit(1)
