"""Extended CPU proof-run for SFT (v2): 200 steps over 250 examples.

Differences vs v1 (scripts/sft_proof_run.py):
  * train_file -> data/curated/proof_train_v2.jsonl (250 examples)
  * max_steps  -> 200 (vs 20)
  * grad_accum -> 4 (effective batch 4)
  * logging_steps -> 5
  * save_steps -> 50
  * lr -> 2e-4 (unchanged)
  * Captures every logged step to outputs/sft/proof_v2/loss_curve.csv
  * Writes outputs/sft/proof_v2/metrics.json with the agreed schema

Stays single-file, mirrors the real backend in train/sft.py.
"""
from __future__ import annotations

import csv
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
    TrainerCallback,
    TrainingArguments,
)

from train.sft import SFTConfig, _load_jsonl, _format_example  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.sft.proof_v2")


class LossCurveCallback(TrainerCallback):
    """Capture loss/lr/grad_norm at every logging step."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.rows: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs:
            return
        # Trainer logs the training loss under "loss".
        if "loss" not in logs:
            return
        row = {
            "step": int(state.global_step),
            "loss": float(logs.get("loss", float("nan"))),
            "lr": float(logs.get("learning_rate", float("nan"))),
            "grad_norm": float(logs.get("grad_norm", float("nan"))),
        }
        self.rows.append(row)
        # Flush each row so partial progress is preserved on crash.
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "grad_norm"])
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


def run(cfg: SFTConfig, dataset: Dataset, max_steps: int, loss_csv: Path) -> dict:
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
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        seed=cfg.seed,
        max_steps=max_steps,
        gradient_checkpointing=False,  # CPU + PEFT: disabled
        optim="adamw_torch",
        bf16=False,
        fp16=False,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    callback = LossCurveCallback(loss_csv)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[callback],
    )

    t0 = time.time()
    res = trainer.train()
    elapsed = time.time() - t0

    n_tokens = sum(len(x["input_ids"]) for x in tokenized)
    # Tokens processed = sum-tokens * (steps * effective-batch / dataset_size), roughly.
    # Stick with the simple "dataset tokens / wall time" denominator as v1 did.
    tok_per_s = n_tokens / max(elapsed, 1e-6)
    log.info("Training done: %.1fs, ~%.0f tok/s", elapsed, tok_per_s)

    model.save_pretrained(str(cfg.output_dir))
    tok.save_pretrained(str(cfg.output_dir))

    final_loss = float(res.training_loss)
    return {
        "kind": "sft",
        "model": "Qwen2.5-0.5B-LoRA-r8",
        "base_model": cfg.model,
        "n_examples": len(dataset),
        "final_loss": final_loss,
        "steps": int(max_steps),
        "lora_rank": int(cfg.rank),
        "lora_alpha": int(cfg.alpha),
        "tokens_per_s": tok_per_s,
        "elapsed_s": elapsed,
    }


def main() -> None:
    output_dir = Path("outputs/sft/proof_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = output_dir / "loss_curve.csv"

    # Honour MAX_STEPS env override so we can drop to 100 if wall-clock blows out.
    max_steps = int(os.environ.get("MAX_STEPS", "200"))

    cfg = SFTConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_file=Path("data/curated/proof_train_v2.jsonl"),
        output_dir=output_dir,
        rank=8,
        alpha=16,
        dropout=0.05,
        epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        grad_accum=1,
        micro_batch=1,
        max_seq_length=384,
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

    metrics = run(cfg, dataset, max_steps=max_steps, loss_csv=loss_csv)
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Metrics: %s", metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("Proof-v2 run failed: %s", exc)
        sys.exit(1)
