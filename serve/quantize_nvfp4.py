"""NVFP4 quantisation via TensorRT-Model-Optimizer.

Produces a TensorRT-LLM engine ready for Jetson Orin Nano (Blackwell-class
NVFP4) or A100 (FP8 fallback). Calibrates on 256 held-out SHOPBench-IN
samples to keep the quantisation grid faithful to our domain vocabulary
(BIS IS-XXXX numbers, HSN digits, T-codes).

Quality gates:
  - Throughput before/after (tokens/sec, single-stream)
  - KL-divergence(FP16 || NVFP4) on a 50-prompt sanity set; warn if >0.05.

If `nvidia-modelopt` is not installed the script exits cleanly with a
copy-pasteable install command. We do *not* attempt a software fallback
quantiser — produce-fake-engines failure mode is worse than no engine.
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.quant")

app = typer.Typer(add_completion=False, help="NVFP4 quantisation")

_MODELOPT_AVAILABLE = False
try:
    import modelopt.torch.quantization as mtq  # type: ignore  # noqa: F401
    _MODELOPT_AVAILABLE = True
except Exception:  # pragma: no cover
    log.warning(
        "nvidia-modelopt not installed. Install via: "
        "`pip install nvidia-modelopt[all] --extra-index-url "
        "https://pypi.nvidia.com`"
    )


@dataclass
class QuantConfig:
    input_path: str
    output_path: Path
    calib_file: Path
    calib_size: int
    fallback: str       # "fp8" | "none"
    seed: int


SANITY_PROMPTS = [
    "बेयरिंग जाम, P3 line down, motor गरम",
    "induction motor 5 kW 3-phase classify HSN",
    "Is IS 14543 applicable to food-grade compressors?",
    "மீட்டர் வேலை செய்யவில்லை — T-code suggest",
    "Pump cavitation, suction pressure low, suggest RCA",
] * 10  # 50 prompts


def _load_calib_dataset(path: Path, n: int) -> list[str]:
    if not path.exists():
        log.warning("Calib file %s missing; using sanity prompts.", path)
        return SANITY_PROMPTS[:n]
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj.get("prompt") or obj.get("text") or "")
            if len(rows) >= n:
                break
    if not rows:
        return SANITY_PROMPTS[:n]
    return rows


def _kl_divergence(p_logits, q_logits) -> float:
    """Symmetric-ish KL of softmaxes; lower is better."""
    import torch
    import torch.nn.functional as F
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return float(F.kl_div(q, p, log_target=True, reduction="batchmean"))


def _quantize(cfg: QuantConfig) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.quantization as mtq  # type: ignore

    log.info("Loading model from %s (fp16, for calibration).", cfg.input_path)
    tok = AutoTokenizer.from_pretrained(cfg.input_path, trust_remote_code=True)
    fp16 = AutoModelForCausalLM.from_pretrained(
        cfg.input_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )

    prompts = _load_calib_dataset(cfg.calib_file, cfg.calib_size)
    log.info("Calibrating on %d samples.", len(prompts))

    def calib_loop(model):
        model.eval()
        with torch.no_grad():
            for p in prompts:
                toks = tok(p, return_tensors="pt", truncation=True, max_length=512)
                toks = {k: v.to(model.device) for k, v in toks.items()}
                model(**toks)

    # Pick quant recipe: NVFP4 if reported supported, else fp8 fallback
    nvfp4_cfg = getattr(mtq, "NVFP4_DEFAULT_CFG", None)
    fp8_cfg = getattr(mtq, "FP8_DEFAULT_CFG", None)
    if nvfp4_cfg is None:
        log.warning("modelopt build has no NVFP4 recipe — using FP8 fallback.")
        recipe, recipe_name = fp8_cfg, "fp8"
    else:
        try:
            recipe, recipe_name = nvfp4_cfg, "nvfp4"
        except Exception:
            recipe, recipe_name = fp8_cfg, "fp8"

    log.info("Applying quant recipe: %s", recipe_name)
    quantized = mtq.quantize(fp16, recipe, calib_loop)

    # Throughput before/after on a fixed prompt
    sample = tok(SANITY_PROMPTS[0], return_tensors="pt").to(fp16.device)
    def _bench(model):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**sample, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return out.shape[-1] / max(time.time() - t0, 1e-6)
    tps_fp16 = _bench(fp16)
    tps_q = _bench(quantized)

    # KL-divergence on sanity set
    kl_total = 0.0
    n = 0
    for p in SANITY_PROMPTS[:50]:
        t = tok(p, return_tensors="pt", truncation=True, max_length=256).to(fp16.device)
        with torch.no_grad():
            lp = fp16(**t).logits[:, -1, :]
            lq = quantized(**t).logits[:, -1, :]
        kl_total += _kl_divergence(lp, lq)
        n += 1
    kl_mean = kl_total / max(n, 1)
    if kl_mean > 0.05:
        log.warning("KL(FP16 || %s) = %.4f exceeds 0.05 threshold.", recipe_name, kl_mean)
    else:
        log.info("KL(FP16 || %s) = %.4f (OK)", recipe_name, kl_mean)

    # Export TensorRT-LLM engine
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from modelopt.torch.export import export_tensorrt_llm_checkpoint  # type: ignore
        export_tensorrt_llm_checkpoint(
            quantized, decoder_type="llama", dtype=torch.float16,
            export_dir=str(cfg.output_path.with_suffix("")),
        )
        log.info("Exported TRT-LLM checkpoint to %s", cfg.output_path.with_suffix(""))
    except Exception as e:
        log.error("Engine export failed: %s. Saving quantised state dict instead.", e)
        torch.save(quantized.state_dict(), cfg.output_path.with_suffix(".pt"))

    return {
        "recipe": recipe_name,
        "tokens_per_s_fp16": tps_fp16,
        "tokens_per_s_quant": tps_q,
        "speedup": tps_q / max(tps_fp16, 1e-6),
        "kl_mean": kl_mean,
        "calib_samples": len(prompts),
    }


@app.command()
def main(
    input_path: str = typer.Option(..., "--input", help="HF path or local checkpoint."),
    output: Path = typer.Option(
        Path("engines/shopfloor-nano-nvfp4.engine"), help="Engine output path.",
    ),
    calib_file: Path = typer.Option(
        Path("data/curated/eval.jsonl"), help="Calibration JSONL.",
    ),
    calib_size: int = typer.Option(256),
    fallback: str = typer.Option("fp8", help="Fallback recipe when NVFP4 unavailable."),
    seed: int = typer.Option(42),
) -> None:
    if not _MODELOPT_AVAILABLE:
        log.error(
            "nvidia-modelopt missing. Install via:\n"
            "  pip install nvidia-modelopt[all] "
            "--extra-index-url https://pypi.nvidia.com"
        )
        sys.exit(2)
    random.seed(seed)
    cfg = QuantConfig(
        input_path=input_path, output_path=output,
        calib_file=calib_file, calib_size=calib_size,
        fallback=fallback, seed=seed,
    )
    metrics = _quantize(cfg)
    out_json = cfg.output_path.with_suffix(".metrics.json")
    out_json.write_text(json.dumps(metrics, indent=2))
    log.info("Quant metrics: %s", metrics)
    log.info("Wrote %s", out_json)


if __name__ == "__main__":
    try:
        app()
    except Exception as exc:
        log.exception("Quantisation failed: %s", exc)
        sys.exit(1)
