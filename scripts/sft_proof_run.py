"""CPU proof-run for the SFT pipeline (HF + PEFT direct).

This is an inline copy of `train.sft._train_hf` with two changes that
matter on a laptop CPU:

  * `gradient_checkpointing=False` — saves no memory at this scale and
    breaks PEFT unless we also call `enable_input_require_grads()`.
    Easier to just turn it off for the 10-step smoke run.
  * `bf16=False, fp16=False` — CPU only.

Everything else (LoRA config, target module discovery, prompt format,
tokenisation, collator, Trainer loop, save_model) is the same as the
real backend in `train/sft.py`.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Reuse the canonical SFTConfig and prompt formatter so this driver
# stays in lock-step with the real trainer.
from train.sft import SFTConfig, _load_jsonl, _format_example  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.sft.proof")


def run(cfg: SFTConfig, dataset: Dataset, max_steps: int = 20) -> dict:
    log.info("Loading tokenizer + model (%s).", cfg.model)
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.float32, trust_remote_code=True,
    )

    present = set()
    for name, _ in model.named_modules():
        for t in cfg.target_modules:
            if name.endswith(t):
                present.add(t)
    if not present:
        log.warning("None of %s found; PEFT picks defaults.", cfg.target_modules)
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
        out = tok(text, truncation=True, max_length=cfg.max_seq_length, padding=False)
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
        logging_steps=1,
        save_steps=10_000,  # don't checkpoint mid-run
        save_total_limit=1,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        seed=cfg.seed,
        max_steps=max_steps,
        gradient_checkpointing=False,  # CPU, tiny model — disabled
        optim="adamw_torch",
        bf16=False,
        fp16=False,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized, data_collator=collator,
    )

    t0 = time.time()
    res = trainer.train()
    elapsed = time.time() - t0

    n_tokens = sum(len(x["input_ids"]) for x in tokenized)
    tok_per_s = n_tokens / max(elapsed, 1e-6)
    log.info("Training done: %.1fs, ~%.0f tok/s", elapsed, tok_per_s)

    # Save the LoRA adapter directly to the output dir (no /final subdir
    # so the file paths in the RECEIPT.md match the spec).
    model.save_pretrained(str(cfg.output_dir))
    tok.save_pretrained(str(cfg.output_dir))

    return {
        "elapsed_s": elapsed,
        "tokens_per_s": tok_per_s,
        "train_loss": float(res.training_loss),
        "steps": max_steps,
        "backend": "hf+peft",
        "model": cfg.model,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
    }


def main() -> None:
    output_dir = Path("outputs/sft/proof")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SFTConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_file=Path("data/curated/proof_train.jsonl"),
        output_dir=output_dir,
        rank=8,
        alpha=16,
        dropout=0.05,
        epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        grad_accum=1,
        micro_batch=1,
        max_seq_length=512,
        seed=42,
        dry_run=False,
        push_to_hub=False,
        hub_repo=None,
        wandb_project="shopfloor-nemotron",
    )
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    rows = _load_jsonl(cfg.train_file)
    log.info("Loaded %d rows from %s", len(rows), cfg.train_file)
    dataset = Dataset.from_list(rows)

    metrics = run(cfg, dataset, max_steps=20)
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Metrics: %s", metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("Proof run failed: %s", exc)
        sys.exit(1)
