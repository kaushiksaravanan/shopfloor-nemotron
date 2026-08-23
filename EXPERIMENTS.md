# EXPERIMENTS — ShopFloor-Nemotron Lab Notebook

> Dated, append-only log of what we tried, what surprised us, and what we plan next.
> One entry per working day. Newest entry on top. Every training run referenced here
> must have a W&B run-id.

Format:

```
## YYYY-MM-DD — short title

**What we tried.**
**What surprised us.**
**Next step.**
```

---

## 2026-07-01 — Day 0: lock SHOPBench-IN spec, seed-set hand-curated

**What we tried.**

- Locked the SHOPBench-IN v0 specification:
  - 1,200 examples, frozen split (800 train-eval / 200 dev / 200 held-out).
  - 6 plant types × 4 languages × 50 BIS standards, stratified.
  - Score = `0.4·ticket_schema_F1 + 0.3·BIS_citation_acc + 0.2·HSN_acc + 0.1·latency_under_2s`.
  - Schema target: SAP-PM IW21 Notification JSON, 11 required fields.
- Hand-curated 60 seed (complaint → ticket) pairs across the 6 plant types.
  Kaushik wrote the 10 CNC-machining and 10 EV-battery-assembly seeds from
  real SAP-PM tickets (anonymised); Achintya wrote the injection-moulding and
  textile-dyeing seeds; Varun wrote the food-processing and pharma-packaging seeds.
- Stood up the empty repo skeleton: pyproject, Makefile, CI, license, this notebook.
- Sanity-loaded `nvidia/Nemotron-Nano-9B-v2` from Hugging Face on a single A100
  in fp16 — generation works, tokenizer round-trips Devanagari and Tamil scripts
  correctly.

**What surprised us.**

- Nemotron 3 Nano's tokenizer is *noticeably* better on Hinglish than on pure
  Devanagari Hindi — code-mixed "spindle garam ho gaya" tokenises in 6 tokens,
  while pure-Devanagari "स्पिंडल गरम हो गया" takes 14. This is *good* news for
  shop-floor input, which is overwhelmingly code-mixed.
- The official model card lists a hybrid Mamba-Transformer architecture. The
  Mamba blocks have no `q_proj`/`k_proj`/`v_proj` modules, so vanilla LoRA
  targeting `["q_proj","k_proj","v_proj","o_proj"]` will only adapt the
  Transformer blocks — the SSM mixers stay frozen. We need to decide whether
  that is enough or whether we adapt the Mamba `in_proj`/`out_proj` too.

**Next step.**

Open questions to answer in week 1:

1. **LoRA rank & target modules for a hybrid Mamba-Transformer.** Sweep
   `rank ∈ {8, 16, 32, 64}` × `{transformer-only, transformer+mamba}` on a
   1k-example subset and pick the Pareto knee on SHOPBench-IN dev score
   vs. trainable-parameter count.
2. **NeMo Gym FastAPI latency at GRPO sample rate.** GRPO will hit the env
   ~256 times per training step. The env evaluates BIS citation, HSN code,
   and SAP-PM schema — each is a sub-second check on its own, but the
   composition matters. Target: p95 < 50 ms per scoring call so the trainer
   does not stall.
3. **NVFP4 distillation loss for the Mamba blocks.** TensorRT-Model-Optimizer
   NVFP4 was validated on dense Transformers. Mamba's selective-scan kernel
   may need a higher per-block calibration sample budget. Measure: NVFP4 vs.
   bf16 KL-divergence on a 256-example calibration set, per layer type.

Run-ids will be linked from this notebook as the sweeps land.

---

## 2026-07-01 — Day 0 addendum: Team Strengths Mapping

Not a lab result, but worth pinning before the sweeps start so reviews route correctly.
Each member's prior production work maps cleanly onto one of the three planes:

**Kaushik → train plane + eval gates (`train/`, `eval/`, `bench/`, `quant/`).**
He has already shipped the closest analogue to what we're doing here: a DeBERTa
finetune on the **Naamapadam** dataset for Indic NER across 11 Indian languages,
hitting 91 F1. That is the same problem shape as the Nemotron LoRA SFT on
Hinglish/Tamil shop-floor input — multilingual subword tokenisation, code-mix,
structured-span outputs. Pair that with his GDPR-compliant enterprise RAG
(94% recall, 2s latency) and the 300k → 2M+ doc HNSW scale-up, and the SHOPBench-IN
eval-harness latency budget (`< 2s` p95 on Jetson) is something he has hit before
on a different stack. He locked the SHOPBench-IN v0 spec on Day 0.

**Achintya → data plane + NeMo Gym (`data/`, `gym/`).**
His **DLM S/4HANA FBE Joule AI Agent Monitor** is, structurally, the production
sibling of this project: a conversational pipeline-monitoring agent that does
root-cause analysis and predictive suggestions on a live SAP pipeline. The
complaint → RCA → SAP-PM ticket flow we're training Nemotron to do is the same
control loop, just pushed offline onto a Jetson. His SAP Invent hackathon piece
(Candidate Verification Engine on SAP BTP, Document AI + LLM) is the synthetic-
data generation pattern we're using in `data/designer/` and `data/curator/`. He
also brings RAG-with-SAP-HANA-Vector-Engine experience for the retrieval
fallback path.

**Varun → serve plane + infra (`serve/`, `scripts/`, `Dockerfile`, CI).**
3 years of **SAP BASIS** administration at Accenture, plus Azure cloud
infrastructure, Linux-based DB ops, system monitoring, and performance tuning.
The NeMo Gym FastAPI service has to hold p95 < 50 ms under GRPO's ~256
calls-per-step load; the NIM-on-A100 fallback needs production-grade health
checks; the Jetson Orin Nano deployment is fundamentally an embedded-Linux
ops problem. All three are squarely in his wheelhouse. He owns the Dockerfile,
the systemd units on the Jetson, and the CI.

**Why this mapping matters for the sweeps.**
The Day 0 open questions (LoRA rank for hybrid Mamba-Transformer, Gym latency
at GRPO sample rate, NVFP4 distillation loss on Mamba blocks) each fall in a
single owner's lane. We won't be hand-offing finetuning decisions to the infra
owner, or vice versa, which keeps reviews fast.
