"""GRPO trainer for ShopFloor-Nemotron.

Reads SFT-merged checkpoint, rolls out groups of K=8 generations against the
FastAPI Gym (`/verify` endpoint), and updates with the GRPO objective.

Preferred backend: NeMo RL (Megatron-RL successor, GRPO native).
Fallback: HF TRL GRPOTrainer for laptops / CI / GPU-less iteration.

Anti reward-hacking
-------------------
The /verify endpoint scores schema/BIS/HSN/T-code/conf_cal separately.
We wrap the reward to penalise:
  - empty-JSON gaming (`{}` or `{"...": null}` payloads),
  - missing BIS citation when the prompt was a compliance ask,
  - extremely short responses (<20 tokens).

These checks happen in `_shape_reward()` and are easy to extend.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.grpo")

# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #
_NEMO_RL_AVAILABLE = False
try:
    import nemo_rl  # type: ignore  # noqa: F401
    _NEMO_RL_AVAILABLE = True
    log.info("NeMo RL backend detected (preferred).")
except Exception:  # pragma: no cover
    log.warning(
        "NeMo RL not installed; using HF TRL fallback. "
        "On the actual training cluster we use NeMo RL. "
        "Install via: `pip install nemo-rl`"
    )

import httpx  # noqa: E402

app = typer.Typer(add_completion=False, help="ShopFloor-Nemotron GRPO RL")

# Default weights mirror configs/grpo.yaml
DEFAULT_WEIGHTS: dict[str, float] = {
    "schema_match": 0.20,
    "bis": 0.30,
    "hsn": 0.30,
    "tcode": 0.10,
    "conf_cal": 0.10,
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GRPOConfig:
    model: str
    gym_url: str
    output_dir: Path
    group_size: int
    kl_coef: float
    learning_rate: float
    max_steps: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    dry_run: bool
    wandb_project: str
    reward_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


# --------------------------------------------------------------------------- #
# Gym client
# --------------------------------------------------------------------------- #
class GymClient:
    """Thin client over the FastAPI verifier env at /reset, /step, /verify."""

    def __init__(self, url: str, timeout_s: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    def reset(self) -> dict[str, Any]:
        r = self._client.post(f"{self.url}/reset")
        r.raise_for_status()
        return r.json()

    def verify(self, prompt: str, response: str) -> dict[str, float]:
        """Return the dict of reward components from the gym."""
        r = self._client.post(
            f"{self.url}/verify", json={"prompt": prompt, "response": response},
        )
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


def _mock_verify(prompt: str, response: str) -> dict[str, float]:
    """Stand-in scorer for --dry-run when no gym is reachable."""
    # crude but deterministic: rewards containing IS-XXXX, HSN digits, T-code.
    parsed_ok = response.strip().startswith("{") and response.strip().endswith("}")
    return {
        "schema_match": 1.0 if parsed_ok else 0.0,
        "bis": 1.0 if "IS " in response else 0.0,
        "hsn": 1.0 if any(c.isdigit() for c in response) else 0.0,
        "tcode": 1.0 if "IW" in response or "IK" in response else 0.0,
        "conf_cal": random.uniform(0.4, 0.9),
    }


# --------------------------------------------------------------------------- #
# Reward shaping
# --------------------------------------------------------------------------- #
def _shape_reward(
    prompt: str,
    response: str,
    components: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Combine weighted components + anti-hacking penalties.

    Returns (final_reward, breakdown_for_logging).
    """
    breakdown = {k: float(v) * weights.get(k, 0.0) for k, v in components.items()}
    base = sum(breakdown.values())

    penalty = 0.0
    stripped = response.strip()
    # 1) Empty/degenerate JSON gaming
    try:
        obj = json.loads(stripped)
        if not obj or all(v in (None, "", []) for v in obj.values()):
            penalty += -1.0
            breakdown["pen_empty_json"] = -1.0
    except Exception:
        pass

    # 2) Compliance prompts must cite BIS
    if any(kw in prompt.lower() for kw in ("bis", "is 14", "compliance", "is14")):
        if components.get("bis", 0.0) < 0.5:
            penalty += -0.3
            breakdown["pen_no_bis"] = -0.3

    # 3) Very short replies (cheap baseline gaming)
    if len(stripped.split()) < 20:
        penalty += -0.5
        breakdown["pen_too_short"] = -0.5

    final = base + penalty
    breakdown["_total"] = final
    return final, breakdown


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _train_nemo_rl(cfg: GRPOConfig, gym: GymClient) -> dict[str, Any]:
    """NeMo RL GRPO. Real training run."""
    log.info(
        "NeMo RL GRPO: group=%d, kl=%.3f, lr=%.0e, steps=%d",
        cfg.group_size, cfg.kl_coef, cfg.learning_rate, cfg.max_steps,
    )
    # Import here so the HF-only fallback doesn't require nemo_rl to import.
    from nemo_rl.algorithms.grpo import GRPOTrainer  # type: ignore

    def reward_fn(prompts: list[str], responses: list[str]) -> list[float]:
        out = []
        for p, r in zip(prompts, responses):
            comps = gym.verify(p, r)
            final, _ = _shape_reward(p, r, comps, cfg.reward_weights)
            out.append(final)
        return out

    trainer = GRPOTrainer(
        model_path=cfg.model,
        group_size=cfg.group_size,
        kl_coef=cfg.kl_coef,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        reward_fn=reward_fn,
        output_dir=str(cfg.output_dir),
    )
    t0 = time.time()
    trainer.train()
    return {"elapsed_s": time.time() - t0, "backend": "nemo_rl"}


def _train_trl(cfg: GRPOConfig, gym: GymClient | None) -> dict[str, Any]:
    """HF TRL fallback (or pure mock when --dry-run)."""
    try:
        from trl import GRPOConfig as TRLGRPOConfig  # type: ignore
        from trl import GRPOTrainer as TRLGRPOTrainer  # type: ignore
        trl_available = True
    except Exception:
        trl_available = False
        if not cfg.dry_run:
            log.error("HF TRL not installed and NeMo RL missing — cannot train.")
            raise

    # ------------------------------------------------------------------- #
    # Mock-only dry run loop: 5 steps, synthetic prompts, mock rewards
    # ------------------------------------------------------------------- #
    if cfg.dry_run:
        log.info("--dry-run: GRPO mock loop, %d steps", cfg.max_steps)
        prompts = [
            "बेयरिंग जाम P3 line down",
            "induction motor 5 kW HSN?",
            "Is this compressor BIS compliant?",
            "மீட்டர் வேலை செய்யவில்லை T-code?",
            "ambiguous failure, more info?",
        ]
        history: list[dict[str, Any]] = []
        for step in range(cfg.max_steps):
            grp_rewards = []
            for _ in range(cfg.group_size):
                prompt = random.choice(prompts)
                fake_resp = json.dumps({
                    "rca": "stub", "bis": "IS 14543", "hsn": "84821010",
                    "tcode": "IW21", "confidence": 0.7,
                })
                comps = _mock_verify(prompt, fake_resp)
                final, breakdown = _shape_reward(
                    prompt, fake_resp, comps, cfg.reward_weights,
                )
                grp_rewards.append(final)
                history.append({"step": step, **breakdown})
            log.info(
                "step %d/%d  mean_reward=%.3f", step + 1, cfg.max_steps,
                sum(grp_rewards) / len(grp_rewards),
            )
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "dry_run_history.json").write_text(
            json.dumps(history, indent=2)
        )
        return {"backend": "mock", "steps": cfg.max_steps}

    # ------------------------------------------------------------------- #
    # Real TRL GRPO path (requires GPU + transformers + trl)
    # ------------------------------------------------------------------- #
    assert trl_available
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def reward_fn(prompts, completions, **kw):  # TRL signature
        rewards = []
        for p, c in zip(prompts, completions):
            comps = gym.verify(p, c) if gym is not None else _mock_verify(p, c)
            final, _ = _shape_reward(p, c, comps, cfg.reward_weights)
            rewards.append(final)
        return rewards

    train_dataset = [{"prompt": p} for p in [
        "बेयरिंग जाम, P3 line down, motor गरम",
        "induction motor 5 kW classify HSN",
        "Compressor BIS compliance?",
    ]]

    args = TRLGRPOConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        num_generations=cfg.group_size,
        max_completion_length=cfg.max_new_tokens,
        temperature=cfg.temperature,
        beta=cfg.kl_coef,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        seed=cfg.seed,
    )
    trainer = TRLGRPOTrainer(
        model=cfg.model,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=train_dataset,
        processing_class=tok,
    )
    t0 = time.time()
    trainer.train()
    return {"elapsed_s": time.time() - t0, "backend": "trl"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@app.command()
def main(
    model: str = typer.Option("outputs/sft/merged", help="SFT checkpoint path."),
    gym_url: str = typer.Option("http://localhost:8000", help="Gym FastAPI URL."),
    output_dir: Path = typer.Option(Path("outputs/grpo")),
    group_size: int = typer.Option(8),
    kl_coef: float = typer.Option(0.04),
    learning_rate: float = typer.Option(1e-6),
    max_steps: int = typer.Option(200),
    max_new_tokens: int = typer.Option(1024),
    temperature: float = typer.Option(0.8),
    top_p: float = typer.Option(0.95),
    seed: int = typer.Option(42),
    dry_run: bool = typer.Option(False, "--dry-run", help="5 steps, mock rewards."),
    wandb_project: str = typer.Option("shopfloor-nemotron"),
) -> None:
    cfg = GRPOConfig(
        model=model,
        gym_url=gym_url,
        output_dir=output_dir,
        group_size=group_size,
        kl_coef=kl_coef,
        learning_rate=learning_rate,
        max_steps=5 if dry_run else max_steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        dry_run=dry_run,
        wandb_project=wandb_project,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    if os.getenv("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    # Probe gym (skip on dry-run)
    gym: GymClient | None = None
    if not dry_run:
        try:
            gym = GymClient(cfg.gym_url)
            gym.reset()
            log.info("Connected to gym at %s", cfg.gym_url)
        except Exception as e:
            log.error("Could not connect to gym at %s: %s", cfg.gym_url, e)
            raise

    try:
        if _NEMO_RL_AVAILABLE and not dry_run:
            metrics = _train_nemo_rl(cfg, gym)  # type: ignore[arg-type]
        else:
            metrics = _train_trl(cfg, gym)
    finally:
        if gym is not None:
            gym.close()

    log.info("GRPO complete: %s", metrics)
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        app()
    except Exception as exc:
        log.exception("GRPO failed: %s", exc)
        sys.exit(1)
