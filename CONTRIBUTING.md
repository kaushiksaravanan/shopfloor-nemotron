# Contributing to ShopFloor-Nemotron

Three people, one shared hackathon clock. These rules keep merges fast and the
final submission demo-ready.

## Ownership

Reviewers map to the prior experience each member actually brings (see `README.md` → Team):

| Area              | Path(s)                                          | Reviewer  | Why this reviewer                                                                              |
|-------------------|--------------------------------------------------|-----------|------------------------------------------------------------------------------------------------|
| SFT + Eval gates  | `train/`, `eval/`, `bench/`, `quant/`            | Kaushik   | LoRA SFT — direct transfer from his DeBERTa + Naamapadam (11-language Indic NER) work; locked the SHOPBench-IN spec. |
| Data + NeMo Gym   | `data/`, `gym/`                                  | Achintya  | His DLM S/4HANA FBE Joule AI Agent Monitor is the RCA-on-SAP-pipeline pattern this project mirrors; Document AI background covers synthetic data generation. |
| Serve + Infra     | `serve/`, `scripts/`, `Dockerfile`, CI workflows | Varun     | 3 years SAP BASIS + Azure + Linux ops — owns NeMo Gym Docker, NIM deploy, Jetson edge stack.   |
| Cross-cutting     | `README`, `Makefile`, top-level infra            | any two of three | Touches every plane; needs broad sign-off.                                              |

A PR touching another area is fine — but the area owner must approve before
merge. No self-merges, ever, even at 2 AM.

## Branch & PR flow

- One feature per branch. Name it `<area>/<short-slug>` (e.g.
  `train/lora-rank-sweep`, `gym/bis-citation-check`).
- Open the PR against `main` early as a draft. Push often.
- Mark "Ready for review" only when CI is green.
- Reviewer merges with **Squash & merge**, keeping the PR title as the commit subject.

## Commit style

```
<area>: <imperative subject under 60 chars>

<optional body wrapped at 80, explaining the *why*, not the *what*>
```

Areas: `train`, `data`, `gym`, `serve`, `eval`, `quant`, `bench`, `infra`,
`docs`, `repo`.

Examples:

```
train: add LoRA rank sweep (8, 16, 32, 64)
gym: enforce p95 < 50ms on BIS citation endpoint
docs: lock SHOPBench-IN scoring formula in README
```

## Pre-commit hooks

Install once:

```
uv run pre-commit install
```

Every commit runs:

- `ruff check --fix`
- `ruff format`
- `pytest -q -m "not slow and not integration"`

A pre-commit failure aborts the commit. Fix and retry — do **not** `--no-verify`.

## PR description requirements

The PR template (`.github/pull_request_template.md`) enforces this, but in
prose: every PR description must contain

- A 1-sentence summary of intent.
- The file/area touched and why.
- For any PR that runs training (SFT, GRPO, eval, quant), a **W&B run-id**
  in the form `wandb: <entity>/<project>/<run-id>`. PRs missing a run-id when
  they should have one will be requested-changes.

## Hard freeze

**T-12 h before submission: hard freeze.** After freeze:

- No new features.
- No dependency bumps.
- Only changes allowed: documentation typos, demo-script tweaks, README quote
  polish, last-mile bug fixes that block the demo.
- `FREEZE_TIMESTAMP` env var is set in CI; the `freeze-check` job will fail
  any commit landed after that timestamp unless its message contains
  `[freeze-exempt]` *and* two reviewers approve.

## What not to commit

- Real customer data — anonymise before committing seeds.
- Model checkpoints — push to Hugging Face Hub, link in the PR.
- W&B API keys, HF tokens, OpenAI keys — they go in `.env`, which is gitignored.
- Anything `> 10 MB` — git-lfs lives outside this hackathon scope.

## Demo-day rule

The `main` branch must be green and `make repro` must succeed end-to-end on
a fresh A100 from a clean clone, every working day from T-3 onward. If you
break it, you fix it before logging off.
