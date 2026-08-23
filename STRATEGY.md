# ShopFloor-Nemotron — Team Strategy Memo

**For:** Team RTX1080Ti (Kaushik, Achintya, Varun)
**Re:** NVIDIA India Agentic AI Open Hackathon, Track B
**Date:** July 1, 2026 — submission day

---

## 1. Team-Project Fit — The Receipts

This is not a stretch project. Every layer of the ShopFloor-Nemotron stack maps 1:1 to something a team member has already shipped in production. Lead with this.

| Stack Layer | Owner | Prior Work That Proves It |
|---|---|---|
| **Nemotron 3 Nano finetune on Hinglish/Tamil complaints → structured tickets** | Kaushik | DeBERTa finetuned on Naamapadam, 91 F1 across 11 Indic languages. Same playbook: multilingual Indic transformer + structured output. |
| **Conversational RCA agent over shop-floor pipeline** | Achintya | DLM S/4HANA FBE Joule AI Agent Monitor — already shipped a conversational pipeline RCA system with predictive suggestions. Architectural twin of our project. |
| **Retrieval over SAP-PM history (notifications, work orders, defect codes)** | Kaushik | Enterprise RAG at 94% recall, 2s E2E latency, scaled 300k → 2M+ docs. SAP HANA Vector Engine RAG already in his portfolio. |
| **DPDP / PII-clean training pipeline for shop-floor data** | Kaushik | GE Healthcare SentinelPII Top-12: DeBERTa + Presidio multilingual PII/PHI redaction. Same redaction stack, same compliance posture. |
| **Agentic orchestration (NeMo Agent Toolkit / tool-calling on top of Nemotron)** | Achintya | MS Agent Framework agent on SAP HANA + Teams Bot via Graph; SAP GenAI Hub document grounding & AI agents. |
| **NeMo Gym Docker, NIM deploy, Jetson edge, S/4HANA integration plane** | Varun | 3 yrs SAP BASIS at Accenture — S/4HANA admin, Azure infra, perf tuning. The deploy owner. |
| **Demo polish, deck, IEEE-grade writeup** | Kaushik | DeployFest 2026 shortlist, 2 IEEE papers, PSG Best Outgoing Student. |

Three SAP Labs engineers building an SAP-PM-shaped product. The domain match is not coincidental — it's our moat.

---

## 2. Pitch Framing — Use These Lines Verbatim

**Opener (deck slide 1 / demo intro):**
> "Our lead has already finetuned DeBERTa on Naamapadam to 91 F1 across 11 Indian languages. We're applying the exact same playbook to Nemotron 3 Nano for Hinglish and Tamil shop-floor complaints. This isn't a research stretch — it's a 1:1 transfer to a new domain."

**Architecture slide / Q&A defense:**
> "The conversational RCA pattern is already running in production — Achintya built the DLM S/4HANA Joule Agent Monitor. We're porting that exact pattern from data pipelines to shop-floor notifications, on Nemotron instead of a hosted model, on a Jetson instead of a cloud cluster."

**Compliance / DPDP question:**
> "Multilingual PII redaction isn't a checkbox for us — Kaushik placed Top-12 at GE Healthcare's SentinelPII with the same DeBERTa+Presidio stack we're using to scrub shop-floor data before it hits the Nemotron Gym."

---

## 3. Risk Reassessment — Honest, Post-LinkedIn

**LOWER risk than originally scoped:**
- **Indic finetuning quality** — Kaushik has done this. 91 F1 on 11 languages is the receipt. Hinglish/Tamil on Nemotron is the same loop.
- **Structured-output reliability** — NER → JSON ticket schema is a refinement of work he already shipped.
- **PII/DPDP compliance** — Already a solved muscle, not a 30-day learn.
- **SAP-PM domain modelling** — Three SAP Labs engineers. Defect codes, notification types, work-order schemas are home turf.
- **Agentic orchestration / RCA UX** — Achintya's Joule Monitor is the prior art.

**HIGHER / under-covered risk — own these now:**
- **NeMo / Nemotron-specific tooling** — The team's transformer chops are on DeBERTa + HF, not NeMo Framework / NeMo Gym / Megatron-style sharding. Varun should own the Docker + NIM deploy path this week; Kaushik should burn a weekend on NeMo Gym tutorials before July 9.
- **Jetson edge perf (INT4/INT8 quant, TensorRT-LLM)** — Nobody on the team has a public Jetson track record. Plan a fallback to NIM-on-laptop demo if Jetson quant slips.
- **Real shop-floor data** — No team member has shop-floor SME access. Synthesise aggressively from public SAP-PM notification corpora + translated complaint templates; flag the synthetic-data caveat honestly in the deck rather than letting a judge surface it.
- **Frontend / live demo polish** — Strong backend team, no dedicated FE. Keep the demo CLI-first or Teams-bot-first (Achintya's lane) — don't promise a slick web UI.

**Net:** the model and compliance risk dropped. Bet the remaining 30 days on NeMo tooling fluency and a believable shop-floor data story.
