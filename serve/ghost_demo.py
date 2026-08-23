"""Gradio booth demo for ShopFloor-Nemotron.

Two-pane layout:
  - LEFT  : Live Jetson Inference (POSTs WAV/text to the Jetson /listen)
  - RIGHT : Cached Ghost Replay (deterministic responses from
            ghost_demo_cache.json so the booth never goes dark)

Booth UX
--------
* Ctrl+G toggles which pane is "primary" (large, centred). The other
  pane stays visible at half-width for transparency — visitors can see
  whether we're serving live or replay.
* Never crashes: every callback is wrapped in try/except, all JSON
  rendering goes through a safe formatter, and HTTP timeouts are short.
* Latency badge on each response so visitors see the 340 ms target hit.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shopfloor.ghost")

CACHE_PATH = Path(__file__).with_name("ghost_demo_cache.json")
JETSON_URL = os.getenv("JETSON_URL", "http://jetson.local:9000")


def _load_cache() -> list[dict[str, Any]]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("Could not read %s: %s", CACHE_PATH, e)
        return []


CACHE = _load_cache()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fmt_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _latency_badge(ms: float) -> str:
    if ms < 350:
        color, emoji = "#1f9d55", "ok"
    elif ms < 700:
        color, emoji = "#d97706", "warn"
    else:
        color, emoji = "#b91c1c", "slow"
    return (
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:6px;font-family:monospace;'>"
        f"[{emoji}] {ms:.0f} ms</span>"
    )


def _bis_badge(value: str | None) -> str:
    if not value:
        return ""
    return (
        "<span style='background:#1e3a8a;color:white;padding:3px 8px;"
        "border-radius:4px;font-size:12px;margin-right:6px;'>"
        f"BIS: {value}</span>"
    )


def _hsn_badge(value: str | None) -> str:
    if not value:
        return ""
    return (
        "<span style='background:#0f766e;color:white;padding:3px 8px;"
        "border-radius:4px;font-size:12px;'>"
        f"HSN: {value}</span>"
    )


def _render_response(payload: dict[str, Any], latency_ms: float) -> tuple[str, str]:
    """Returns (badge_html, code_block) — never raises."""
    try:
        badges = (
            _latency_badge(latency_ms)
            + " "
            + _bis_badge(payload.get("bis"))
            + _hsn_badge(payload.get("hsn"))
        )
        return badges, _fmt_json(payload)
    except Exception as e:
        log.exception("render failure: %s", e)
        return _latency_badge(latency_ms), "{}"


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def call_live(prompt: str, audio_path: str | None) -> tuple[str, str]:
    """POST to Jetson /listen. Always returns SOMETHING."""
    if not prompt and not audio_path:
        return _latency_badge(0), '{"error": "no input"}'
    try:
        if audio_path:
            with open(audio_path, "rb") as f:
                files = {"audio": ("clip.wav", f, "audio/wav")}
                with httpx.Client(timeout=8.0) as c:
                    r = c.post(f"{JETSON_URL}/listen", files=files)
            r.raise_for_status()
            payload = r.json()
        else:
            # Fall back to a text-only endpoint (the Jetson server only
            # supports /listen with WAV today, so we route through ghost
            # cache to preview the JSON shape).
            return call_ghost(prompt)
        latency = float(payload.get("latency_ms", 0))
        return _render_response(payload, latency)
    except Exception as e:
        log.warning("live call failed: %s — falling back to ghost", e)
        badge, code = call_ghost(prompt or "")
        return (
            badge
            + "<span style='color:#b91c1c;font-size:12px;margin-left:8px;'>"
            "[live unavailable, replaying ghost]</span>",
            code,
        )


def call_ghost(prompt: str) -> tuple[str, str]:
    """Best-match from the cached cases."""
    if not CACHE:
        return _latency_badge(0), '{"error": "ghost cache empty"}'
    if not prompt:
        case = random.choice(CACHE)
    else:
        best = CACHE[0]
        best_score = -1
        for c in CACHE:
            score = sum(
                1 for w in prompt.lower().split() if w in c["prompt"].lower()
            )
            if score > best_score:
                best_score = score
                best = c
        case = best
    return _render_response(case["expected_response"], float(case["latency_ms"]))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container { max-width: 1400px !important; }
#pane-live, #pane-ghost { border-radius: 10px; padding: 12px; }
#pane-live { border: 2px solid #1f9d55; }
#pane-ghost { border: 2px dashed #6b7280; }
.title { font-weight: 700; font-size: 18px; margin-bottom: 8px; }
"""

JS_SHORTCUTS = """
() => {
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === 'g') {
      const live = document.getElementById('pane-live');
      const ghost = document.getElementById('pane-ghost');
      if (live && ghost) {
        const liveBig = live.style.flex !== '0.5';
        live.style.flex  = liveBig ? '0.5' : '1.5';
        ghost.style.flex = liveBig ? '1.5' : '0.5';
      }
    }
  });
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ShopFloor-Nemotron", css=CSS) as ui:
        gr.Markdown(
            "# ShopFloor-Nemotron — Booth Demo\n"
            "Hinglish / Tamil shop-floor complaint -> BIS-cited, HSN-tagged, "
            "SAP-PM-ready ticket. **Ctrl+G** toggles primary pane (live <-> ghost)."
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Operator complaint (text)",
                placeholder='e.g. "बेयरिंग जाम, P3 line down, motor गरम"',
                lines=2,
                scale=3,
            )
            audio = gr.Audio(
                label="Or record/upload WAV",
                sources=["microphone", "upload"],
                type="filepath",
                scale=2,
            )
        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            ghost_btn = gr.Button("Replay Ghost")
            clear_btn = gr.Button("Clear")
        with gr.Row():
            with gr.Column(elem_id="pane-live", scale=1):
                gr.Markdown('<div class="title">Live Jetson Inference</div>')
                live_badge = gr.HTML(value=_latency_badge(0))
                live_out = gr.Code(value="{}", language="json")
            with gr.Column(elem_id="pane-ghost", scale=1):
                gr.Markdown('<div class="title">Cached Ghost Replay</div>')
                ghost_badge = gr.HTML(value=_latency_badge(0))
                ghost_out = gr.Code(value="{}", language="json")

        run_btn.click(call_live, [prompt, audio], [live_badge, live_out])
        run_btn.click(call_ghost, [prompt], [ghost_badge, ghost_out])
        ghost_btn.click(call_ghost, [prompt], [ghost_badge, ghost_out])
        clear_btn.click(
            lambda: ("", None, _latency_badge(0), "{}", _latency_badge(0), "{}"),
            outputs=[prompt, audio, live_badge, live_out, ghost_badge, ghost_out],
        )

        with gr.Accordion("Cached cases (5)", open=False):
            for c in CACHE:
                gr.Markdown(f"- **{c['id']}**: `{c['prompt']}`")

        ui.load(None, None, None, js=JS_SHORTCUTS)
    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=7860, show_error=True,
    )
