# CLAUDE.md

Project context for an AI coding assistant. Read this before writing or editing code.

---

## Project

**Synthetic Q&A data-generation pipeline for a Home DIY Repair assistant.**

The pipeline generates DIY repair Q&A items with a weak/cheap LLM, gates them on
quality, labels them on 6 dimensions with **both** a human CLI reviewer **and** an
independent LLM-as-Judge, analyzes where quality breaks down by segment, and then
runs a two-phase iteration loop that (A) calibrates the judge against the human, then
(B) corrects the generation prompt — proving a **measurable, data-driven** drop in
failure rate.

This mirrors a real MLOps loop: **generate → evaluate → diagnose → fix.** The
deliverable is not a clean pipeline; it is a *provably better* one (before/after).

---

## What "done" looks like (success criteria)

- ≥ 50 validated Q&A items per run; all 5 categories ≥ 20% of the set (±tolerance).
- ≥ 95% of generated items pass Pydantic structural validation.
- Human labels on ≥ 20 items; LLM-judge labels on every item.
- **Phase A:** human/LLM agreement ≥ 80% on **every** dimension.
- **Phase B:** baseline overall failure ≥ 15%; post-correction failure ≤ 20% of
  baseline (i.e. **> 80% reduction**). Corrected set ≥ 80% overall pass rate.
- Every prompt change documented in the Iteration Log (≥ 4 entries) and traceable to
  a specific segment × dimension — never to intuition.

---

## Model strategy (deliberate, cost-aware split)

Two providers, two keys, **one role-aware client**. Both are OpenAI-compatible, so
both run through `instructor.from_openai(...)` with different base URLs.

| Role | Model | Access | Temp | Why |
|------|-------|--------|------|-----|
| **Generator** | `meta-llama/llama-3.1-8b-instruct` | **OpenRouter** (`https://openrouter.ai/api/v1`) | higher (diversity) | Cheap, deliberately *weak* — gives a real baseline to improve |
| **Judge** | `deepseek-v4-flash` | DeepSeek API (OpenAI-compatible base URL) | low (determinism) | Stronger, different family → avoids self-preference bias |

**Rules that protect the showcase — do not violate:**

1. **Use the explicit model ID `deepseek-v4-flash`, NOT `deepseek-chat`.** The
   `deepseek-chat` alias is deprecating (errors after 2026-07-24). (Upgrade option:
   `deepseek-v4-pro` if a stronger judge anchor is wanted; pick on purpose.)
2. **Vary ONE knob at a time.** Phase A varies the *judge prompt* with the judge model
   fixed. Phase B varies the *generator prompt* with the generator model fixed and the
   judge fully frozen. The before/after numbers are only creditable if nothing else moved.
3. **Never "fix" a Phase B failure by upgrading the generator model.** If the 8B can't
   produce valid JSON often enough, that is a Step 1/Step 2 *structural* problem (solve
   with constrained decoding + retries, logged as its own iteration) — not a Phase B lever.

**OpenRouter caveat:** OpenRouter routes to whichever underlying provider is
available, and structured-output support for an 8B model varies by provider. Lean on
Instructor's **JSON mode + retries** rather than assuming tool-calling works; optionally
pin provider preferences via `extra_body`. Verify the current model slug and pricing on
openrouter.ai before a long run.

---

## Tech stack

- **Python 3.10+**
- **Pydantic** — all structured data (Q&A schema, label record, trace record)
- **Instructor** — schema-safe LLM output for both generator and judge
- **OpenRouter + DeepSeek** — via the OpenAI SDK (both OpenAI-compatible)
- **Logfire** — runtime observability of LLM calls (latency, tokens, retries, errors)
- **datasets** (Hugging Face) — load benchmark for the Step 2 distribution reference
- **rapidfuzz** — normalized-string dedup in Step 2 (embeddings are overkill at this scale)
- **pandas** — aggregation in Step 5
- **matplotlib + seaborn** — the 5 required charts (seaborn heatmap for the segment diagnostic)
- **python-dotenv** — API keys

**Hard rule:** no hardcoded repair answers. All content is LLM-generated at runtime.

---

## Two meanings of "trace" — keep them separate

- **Data trace records** = the spec's per-item audit log spanning Steps 1–4 (which
  prompt made the item, did it clear the gate, human label, judge label, agreement).
  Stored as **JSONL**. This is data lineage.
- **Logfire traces** = runtime spans for the LLM calls themselves. Lives almost entirely
  inside `llm_client.py` (`logfire.configure()` + `instrument_openai()` +
  `instrument_pydantic()`). Attach `trace_id` and step name as span attributes so an
  operational span can be correlated back to its data trace record.

---

## Module layout

```yaml
diy_pipeline/
  config.py            # categories, thresholds, model registry (by role), prompt-version registry
  schemas.py           # ALL Pydantic models (QAItem, LabelRecord, TraceRecord)
  llm_client.py        # role-aware Instructor client + retries + Logfire instrumentation
  prompts.py           # versioned generator + judge prompt templates (keyed by version)
  trace.py             # per-item trace record assembly + JSONL read/write
  benchmark.py         # loads HF dataset, computes reference category distribution

  step1_generate.py    # category-parameterized generation -> QAItem + trace record
  step2_quality_gate.py# structural + per-dim pre-checks (drop) + dedup + distribution (regen)
  step3_human_label.py # CLI: print item, collect 6 pass/fail -> LabelRecord(labeler=human)
  step4_llm_judge.py   # judge: same 6 dims, low temp -> LabelRecord(labeler=llm_judge)
  step5_analysis.py    # segment metrics: per-dim pass rate, human/LLM agreement
  step5_viz.py         # the 5 charts -> visualizations/*.png
  step6_iterate.py     # Phase A (judge) then Phase B (generator) orchestration

  pipeline.py          # CLI orchestrator; each step independently runnable
  data/                # *.jsonl generated data, label files, trace records
  visualizations/      # *.png
  ITERATION_LOG.md     # >= 4 structured entries
  README.md
```

Build foundation (`config`, `schemas`, `llm_client`, `trace`) first, then prove a thin
vertical slice (one category, one dimension, generate → judge → print) before going wide.

---

## Data schema (QAItem — 7 fields)

| Field | Type | Validation |
| ------- | ------ | ------------ |
| `question` | str | non-empty |
| `answer` | str | non-empty |
| `equipment_problem` | str | non-empty |
| `tools_required` | list[str] | ≥ 1 |
| `steps` | list[str] | ≥ 3 |
| `safety_info` | str | non-empty |
| `tips` | list[str] | ≥ 1 |

**Label record** (identical shape for human and judge): `trace_id`, `labeler`, the 6
binary dimension fields, `overall_pass` (= all 6 pass). Judge record adds
`judge_prompt_version`.

---

## The 6 quality dimensions (LLM-judge thresholds)

| # | Dimension | Pass means | Target |
| --- | ----------- | ----------- | -------- |
| D1 | Answer Completeness | enough to complete the repair end to end | ≥ 85% |
| D2 | Safety Specificity | names the specific hazard AND the specific precaution | ≥ 90% |
| D3 | Tool Realism | homeowner-owned / <$50 hardware-store tools only | ≥ 95% |
| D4 | Scope Appropriateness | within DIY capability, or clearly says "call a pro" | ≥ 95% |
| D5 | Context Clarity | answer directly addresses the stated problem | ≥ 90% |
| D6 | Tip Usefulness | non-obvious, task-specific advice beyond the steps | ≥ 85% |

Overall pass = all 6 pass on the same item (target ≥ 80%).

---

## Pipeline steps

1. **Generate** — loop the 5 categories with a category-parameterized prompt; coerce to
   `QAItem`; write JSONL + trace record (prompt variant, category, timestamp, model, raw response).
2. **Quality gate** — (a) Pydantic structural check; (b) cheap per-dim heuristic
   pre-checks (e.g. `safety_info` length, generic-phrase block-list, tool-string
   block-list) — *fast filters, NOT the real judge*; (c) batch dedup + category
   distribution vs benchmark. Drop on per-item fail; regenerate on batch fail.
3. **Human label (CLI)** — walk each item, collect 6 binary labels. ≥ 20 items.
4. **LLM judge** — same 6 dims, lower temp, structured output, judge-prompt version recorded.
5. **Analysis & viz** — join human + judge + gate results into segment metrics
   (category × variant); produce all 5 charts.
6. **Iterate** — Phase A: raise judge↔human agreement to ≥ 80% on every dim (re-run
   Step 4 only). Phase B: with the trusted judge, fix worst segment × worst dimension in
   the generator prompt; re-run Steps 1–5; compute improvement ratio.

The 5 charts (saved as PNG in `visualizations/`): per-dimension pass rate (before/after),
**segment heatmap (segments × 6 dims)**, human-vs-LLM agreement (before/after Phase A),
category distribution vs benchmark, before/after paired bars per dimension.

---

## Structural validity vs content quality (important distinction)

Llama 3.1-8B is shaky on schemas with lists, and `QAItem` has three. Separate the concerns:

- **Structural validity** (parseable, schema-conforming) → protect with constrained
  decoding / JSON mode + Instructor `max_retries`. Target ≥ 95%. This is a plumbing
  problem, fix it in Steps 1–2.
- **Content quality** (specific safety, non-obvious tips) → this is what we *want* to be
  weak at baseline. The 8B will oblige and push baseline failure above the 15% floor
  naturally. This is what Phase B improves.

---

## Showcase design levers (intentional weaknesses)

- **Deliberately loose initial judge prompt** on a subtle dimension (D5 Context Clarity
  or D6 Tip Usefulness) so Phase A has real disagreement to calibrate. An airtight judge
  on attempt one leaves nothing to show.
- **Deliberately weak baseline generator** (cheap 8B + plain prompt) so Phase B has a
  measurable gap to close.

These are features, not bugs — they create the before/after story the project is graded on.

---

## Intended CLI

```bash
python -m diy_pipeline.pipeline generate --n 50 --variant baseline
python -m diy_pipeline.pipeline gate
python -m diy_pipeline.pipeline label-human --n 20
python -m diy_pipeline.pipeline judge --judge-version v1
python -m diy_pipeline.pipeline analyze
python -m diy_pipeline.pipeline iterate --phase a      # re-runs judge only
python -m diy_pipeline.pipeline iterate --phase b      # re-runs gen->analyze with corrected prompt
```

Each step is independently runnable and reads/writes files, so Phase A re-runs Step 4
alone and Phase B re-runs Steps 1–5 without touching the baseline data.

---

## Iteration log format (ITERATION_LOG.md)

```yaml
### Iteration N: [title]
- Date:
- Phase: Baseline | A | B
- Change: what changed vs previous iteration
- Hypothesis: why it should help
- Result: quantitative (min agreement, overall failure rate)
- Decision: keep / revert / modify
- Next step:
```

---

## Conventions & guardrails

- Prompts live in `prompts.py` keyed by version — never inline string-edit a prompt.
- Generator temp > judge temp, always.
- Every LLM call wrapped for rate-limit + malformed-response retry; never crash on one bad output.
- Save baseline and corrected outputs under distinct filenames — both are needed for comparison.
- Don't trust a sample < 30 items; failure rates are noisy at small scale.
- Keys in `.env`, never committed.

---

## Environment

```yaml
OPENROUTER_API_KEY=...     # generator (Llama 3.1-8B)
DEEPSEEK_API_KEY=...        # judge (DeepSeek V4 Flash)
LOGFIRE_TOKEN=...           # observability (optional locally)
```
