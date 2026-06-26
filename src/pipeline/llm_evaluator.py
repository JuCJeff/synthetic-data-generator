"""
LLM-as-Judge labeling.

Scores every validated record on the same 6 binary dimensions the human
labeler uses, via DeepSeek (low temperature) + Instructor structured output.
Appends one labelled record per item to JSONL after each call so progress
survives an early exit.

Resumable: already-judged trace_ids are skipped automatically.

Run:
    uv run python -m src.llm_evaluator
"""

import os
import sys

import instructor
import logfire
from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    DEEPSEEK_BASE_URL,
    JUDGE_MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT_V1,
    JUDGE_TEMPERATURE,
    LABELS_DIR,
    PROJECT_ROOT,
    VALIDATED_OUTPUTS_PATH,
)
from src.instrumentation import configure_logfire
from src.schemas import (
    EVALUATION_DIMENSIONS,
    EvaluationResults,
    LabelledRecord,
    QARecord,
    RepairQA,
)
from src.ui import console, make_progress_bar, print_error
from src.util import append_jsonl, load_labeled_ids, load_qa_records

LLM_JUDGE_LABELS_PATH = LABELS_DIR / "llm_judge_labels.jsonl"

load_dotenv()

_openai_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=DEEPSEEK_BASE_URL,
)
configure_logfire(_openai_client)

# deepseek-v4-flash runs in thinking mode, which rejects forced tool_choice —
# Instructor's default TOOLS mode 400s, so use JSON mode instead
_client = instructor.from_openai(_openai_client, mode=instructor.Mode.JSON)


# --- Prompt ---


def _build_judge_user_prompt(item: RepairQA) -> str:
    dimensions = "\n".join(
        f"{i}. {ed['dimension']}: {ed['requirement']}"
        for i, ed in enumerate(EVALUATION_DIMENSIONS, 1)
    )
    return (
        "Evaluate this DIY repair Q&A item:\n\n"
        f"{item.model_dump_json(indent=2)}\n\n"
        "Score each dimension 1 (pass) or 0 (fail):\n"
        f"{dimensions}"
    )


# --- Judging ---


def _judge_one(record: QARecord) -> LabelledRecord | None:
    """Judge one item. Returns None on failure — never crashes the batch."""
    with logfire.span(
        "judge_one",
        trace_id=record.trace_id,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
    ):
        try:
            results = _client.chat.completions.create(
                model=JUDGE_MODEL,
                response_model=EvaluationResults,
                max_retries=JUDGE_MAX_RETRIES,
                temperature=JUDGE_TEMPERATURE,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT_V1},
                    {
                        "role": "user",
                        "content": _build_judge_user_prompt(record.record),
                    },
                ],
            )
        except Exception as error:
            logfire.warning(
                "Judge call failed", trace_id=record.trace_id, error=str(error)
            )
            return None

        return LabelledRecord(
            trace_id=record.trace_id,
            labeler="llm_judge",
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            results=results,
        )


def judge_items() -> None:
    """Judge every validated record that does not already have an LLM label."""
    records, _ = load_qa_records(VALIDATED_OUTPUTS_PATH)
    if not records:
        print_error(
            "Pipeline Dependency Missing",
            f"No validated records found at {VALIDATED_OUTPUTS_PATH.relative_to(PROJECT_ROOT)}",
            suggestion="Run the validator first: uv run python -m src.validator",
        )
        sys.exit(1)

    already_judged = load_labeled_ids(LLM_JUDGE_LABELS_PATH)
    queue = [r for r in records if r.trace_id not in already_judged]

    if not queue:
        console.print(
            f"[success]All {len(records)} validated items already judged.[/success]"
        )
        return

    console.print(
        f"Judging {len(queue)} of {len(records)} items "
        f"({len(already_judged)} already done) with {JUDGE_MODEL} "
        f"(prompt {JUDGE_PROMPT_VERSION})"
    )

    judged_count = 0
    failed_count = 0

    with make_progress_bar() as progress:
        bar = progress.add_task("Judging items", total=len(queue))
        for record in queue:
            trace_id = record.trace_id
            progress.update(bar, description=trace_id)
            label = _judge_one(record)
            if label is None:
                console.print(f"{trace_id} failed to be judged")
                failed_count += 1
            else:
                append_jsonl(label, LLM_JUDGE_LABELS_PATH)
                judged_count += 1
            progress.advance(bar)

    console.print(
        f"Judged [success]{judged_count}[/success], failed [error]{failed_count}[/error] "
        f"→ [cyan]{LLM_JUDGE_LABELS_PATH}[/cyan]"
    )


if __name__ == "__main__":
    judge_items()
