"""On-Jetson inference wrapper.

Loads a TensorRT-LLM engine + Whisper.cpp via subprocess, exposes a
FastAPI `/listen` endpoint that accepts a WAV upload and returns a
structured JSON ticket. On engine load failure (or with `--ghost-mode`)
it serves the pre-recorded responses from `ghost_demo.py` so the booth
never dies.

This file is meant to run *on* the Jetson Orin Nano. The HF/CPU dev path
uses the same module but instantiates `_StubRunner`, which lets a
teammate hit `/listen` from a laptop and get a plausible response.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.jetson")

app = typer.Typer(add_completion=False, help="Jetson inference server")

# --------------------------------------------------------------------------- #
# Optional imports
# --------------------------------------------------------------------------- #
_TRTLLM_AVAILABLE = False
try:
    from tensorrt_llm.runtime import ModelRunner  # type: ignore
    _TRTLLM_AVAILABLE = True
except Exception:  # pragma: no cover - only on Jetson
    log.info("tensorrt-llm not importable (expected off-Jetson). Using stub runner.")


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
class _StubRunner:
    """Deterministic stand-in: returns ghost-cache best match by keyword."""

    def __init__(self, cache_path: Path) -> None:
        self.cache: list[dict[str, Any]] = []
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        log.info("StubRunner loaded %d cached cases.", len(self.cache))

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not self.cache:
            return json.dumps({"error": "no cache available"})
        # naive keyword overlap
        best = self.cache[0]
        best_score = -1
        for c in self.cache:
            score = sum(1 for w in prompt.lower().split() if w in c["prompt"].lower())
            if score > best_score:
                best_score = score
                best = c
        return json.dumps(best["expected_response"], ensure_ascii=False)


class _TRTLLMRunner:
    """Real TRT-LLM ModelRunner wrapper."""

    def __init__(self, engine_dir: str) -> None:
        from transformers import AutoTokenizer
        self.runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0)
        # Tokenizer is bundled alongside the engine by export_tensorrt_llm_checkpoint
        self.tok = AutoTokenizer.from_pretrained(engine_dir, trust_remote_code=True)
        log.info("TRT-LLM engine loaded from %s", engine_dir)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        ids = self.tok(prompt, return_tensors="pt").input_ids
        out = self.runner.generate(
            batch_input_ids=[ids[0]], max_new_tokens=max_new_tokens,
            end_id=self.tok.eos_token_id, pad_id=self.tok.pad_token_id,
            temperature=0.0,
        )
        gen_ids = out[0][0][len(ids[0]):]
        return self.tok.decode(gen_ids, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Whisper ASR
# --------------------------------------------------------------------------- #
def _whisper_transcribe(wav_path: Path) -> str:
    """Shell out to whisper.cpp; returns transcript string."""
    binary = os.getenv("WHISPER_BIN", "whisper-cpp")
    model = os.getenv("WHISPER_MODEL", "models/ggml-small.bin")
    try:
        out = subprocess.check_output(
            [binary, "-m", model, "-f", str(wav_path), "-l", "auto", "-otxt", "-nt"],
            stderr=subprocess.STDOUT, timeout=15,
        )
        return out.decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        log.warning("whisper-cpp binary not found at %s; returning placeholder.", binary)
        return "[whisper unavailable]"
    except subprocess.CalledProcessError as e:
        log.error("whisper failed: %s", e.output.decode("utf-8", errors="replace"))
        return ""


# --------------------------------------------------------------------------- #
# SAP S/4HANA poster
# --------------------------------------------------------------------------- #
def _post_sap(ticket: dict[str, Any]) -> dict[str, Any]:
    """POST the ticket to S/4HANA PM. No-op when SAP_PM_URL is unset."""
    url = os.getenv("SAP_PM_URL")
    if not url:
        return {"posted": False, "reason": "SAP_PM_URL unset"}
    try:
        import httpx
        with httpx.Client(timeout=5.0) as c:
            r = c.post(url, json=ticket)
        return {"posted": r.status_code < 300, "status": r.status_code}
    except Exception as e:
        log.warning("SAP POST failed: %s", e)
        return {"posted": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# FastAPI app builder
# --------------------------------------------------------------------------- #
def _build_app(runner, ghost_mode: bool):
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse

    api = FastAPI(title="ShopFloor-Nemotron Jetson")

    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "ghost_mode": ghost_mode, "runner": type(runner).__name__}

    @api.post("/listen")
    async def listen(audio: UploadFile = File(...)) -> JSONResponse:
        t0 = time.time()
        # 1) save WAV temp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await audio.read())
            wav_path = Path(tmp.name)

        # 2) ASR
        t_asr0 = time.time()
        transcript = _whisper_transcribe(wav_path)
        t_asr = (time.time() - t_asr0) * 1000

        # 3) Generate
        t_gen0 = time.time()
        try:
            raw = runner.generate(transcript, max_new_tokens=256)
        except Exception as e:
            log.exception("Generation failed: %s", e)
            raw = json.dumps({"error": str(e)})
        t_gen = (time.time() - t_gen0) * 1000

        # 4) Parse + POST
        try:
            ticket = json.loads(raw)
        except json.JSONDecodeError:
            ticket = {"raw": raw}
        t_post0 = time.time()
        sap_result = _post_sap(ticket)
        t_post = (time.time() - t_post0) * 1000

        total = (time.time() - t0) * 1000
        body = {
            "transcript": transcript,
            **(ticket if isinstance(ticket, dict) else {"output": ticket}),
            "latency_ms": round(total, 1),
            "timings": {
                "asr_ms": round(t_asr, 1),
                "gen_ms": round(t_gen, 1),
                "sap_ms": round(t_post, 1),
            },
            "sap": sap_result,
            "ghost_mode": ghost_mode,
        }
        log.info("listen done in %.1f ms (asr=%.1f gen=%.1f post=%.1f)",
                 total, t_asr, t_gen, t_post)
        try:
            wav_path.unlink()
        except OSError:
            pass
        return JSONResponse(body)

    return api


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@app.command()
def main(
    engine: str = typer.Option(
        "/opt/shopfloor/shopfloor-nano-nvfp4", help="TRT-LLM engine dir.",
    ),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(9000),
    ghost_mode: bool = typer.Option(False, "--ghost-mode"),
    ghost_cache: Path = typer.Option(Path("serve/ghost_demo_cache.json")),
) -> None:
    import uvicorn

    runner: Any
    if ghost_mode or not _TRTLLM_AVAILABLE:
        if not ghost_mode:
            log.warning("TRT-LLM unavailable — forcing ghost mode.")
        runner = _StubRunner(ghost_cache)
        ghost_mode = True
    else:
        try:
            runner = _TRTLLMRunner(engine)
        except Exception as e:
            log.error("Engine load failed (%s) — flipping to ghost mode.", e)
            runner = _StubRunner(ghost_cache)
            ghost_mode = True

    api = _build_app(runner, ghost_mode=ghost_mode)
    log.info("Listening on http://%s:%d (ghost_mode=%s)", host, port, ghost_mode)
    uvicorn.run(api, host=host, port=port, log_level="info")


if __name__ == "__main__":
    try:
        app()
    except Exception as exc:
        log.exception("jetson_inference failed: %s", exc)
        sys.exit(1)
