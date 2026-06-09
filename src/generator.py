"""
Builds a flat generation plan from the taxonomy in schemas.py, then executes
one LLM call per task using Instructor + OpenRouter (mistral-small-3.2-24b). The LLM
picks a specific subcategory from a provided menu on each call. Validates output
against the RepairQA Pydantic schema and saves results to JSONL.

Run:
    uv run python -m src.generator
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import instructor
import logfire
from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    GENERATION_MODEL_V1,
    GENERATION_TEMPERATURE_V1,
    MAX_RETRIES_V1,
    OPENROUTER_BASE_URL,
    PROJECT_ROOT,
    SYSTEM_PROMPT_V1,
)
from src.instrumentation import configure_logfire
from src.schemas import (
    CATEGORY_SUBCATEGORIES,
    GeneratedRecord,
    GenerationTask,
    RepairQA,
)
from src.ui import (
    console,
    make_progress_bar,
    print_batch_summary,
    print_generation_error,
)
from src.util import generate_uuid, save_jsonl

_GENERATION_DIR = PROJECT_ROOT / "data" / "generated"
_GENERATION_OUTPUT_PATH = _GENERATION_DIR / "batch_v1.jsonl"
_CACHE_DIR = _GENERATION_DIR / "cache"

_PROMPT_VARIANT = "v1"

load_dotenv()


_openai_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
)
configure_logfire(_openai_client)

# Wrap it with Instructor for structured output
_client = instructor.from_openai(_openai_client)


# --- user prompt factory ---
# User prompt factory
def _build_generation_variant_hint(variant: int) -> str:
    """Encourage the LLM to pick differently across variants for the same category."""
    if variant == 0:
        return "This is the first item for this category."
    return (
        f"This is item #{variant + 1} for this category. "
        f"Pick a DIFFERENT subcategory or specific problem than a typical first choice. "
        f"Vary the equipment and scenario."
    )


def _build_user_prompt(
    category: str,
    subcategory_options: list[str],
    variant: int,
) -> str:
    options = ", ".join(subcategory_options)
    return (
        f"Category: {category}\n"
        f"Choose one subcategory from this list to focus on: {options}\n"
        f"{_build_generation_variant_hint(variant)}\n"
        f"Generate a complete, realistic DIY repair Q&A item."
    )


# --- Caching ---
def _hash_prompt(task: GenerationTask) -> str:
    raw = f"{task.category}|{task.variant}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _save_generation_to_cache(prompt_hash: str, item: RepairQA) -> None:
    """Save a generated item to disk so future runs can skip the LLM call."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{prompt_hash}.json"
    path.write_text(item.model_dump_json(indent=2))


def _load_generation_from_cache(prompt_hash: str) -> RepairQA | None:
    path = _CACHE_DIR / f"{prompt_hash}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return RepairQA.model_validate(data)
    except Exception as error:
        logfire.warning("Cache read failed, skipping", path=str(path), error=str(error))
        return None


# --- Planning ---
def _generate_repair_qa(
    category: str,
    subcategory_options: list[str],
    variant: int,
) -> RepairQA | None:
    """
    Generate one validated RepairQA item via Instructor + OpenRouter.
    Instructor enforces the RepairQA schema and auto-retries on validation failure.
    """
    try:
        return _client.chat.completions.create(
            model=GENERATION_MODEL_V1,
            response_model=RepairQA,
            max_retries=MAX_RETRIES_V1,
            temperature=GENERATION_TEMPERATURE_V1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V1},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        category, subcategory_options, variant
                    ),
                },
            ],
        )
    except Exception as e:
        print_generation_error(category, e)


def build_generation_plan(items_per_category: int = 10) -> list[GenerationTask]:
    """
    Flatten category × variant into a list of tasks.
    With items_per_category=10: 5 categories × 10 variants = 50 tasks total.
    """
    tasks: list[GenerationTask] = []

    for category in CATEGORY_SUBCATEGORIES.keys():
        for variant in range(items_per_category):
            tasks.append(
                GenerationTask(
                    category=category,
                    variant=variant,
                )
            )

    return tasks


# --- Execution ---


def _generate_one_repair_qa(task: GenerationTask) -> GeneratedRecord | None:
    """Generate one item from a planned task. Uses disk cache to skip repeat LLM calls."""
    with logfire.span(
        "generate_one",
        category=task.category,
        variant=task.variant,
    ):
        prompt_hash = _hash_prompt(task)

        item = _load_generation_from_cache(prompt_hash)

        if item is not None:
            logfire.info("Cache hit", category=task.category, prompt_hash=prompt_hash)
        else:
            item = _generate_repair_qa(
                category=task.category,
                subcategory_options=CATEGORY_SUBCATEGORIES[task.category],
                variant=task.variant,
            )
            if item is None:
                logfire.warning(
                    "Generation failed",
                    category=task.category,
                    variant=task.variant,
                )
                return None
            _save_generation_to_cache(prompt_hash, item)

        return GeneratedRecord(
            trace_id=f"qa_{generate_uuid()}",
            category=task.category,
            subcategory=item.chosen_subcategory,
            prompt_variant=_PROMPT_VARIANT,
            prompt_hash=prompt_hash,
            model_used=GENERATION_MODEL_V1,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            record=item,
        )


def generate_qa_dataset(items_per_category: int = 10) -> list[GeneratedRecord]:
    """Build the plan, then execute it. Returns validated records."""
    logfire.info(
        "Starting baseline generation run",
        items_per_category=items_per_category,
        model=GENERATION_MODEL_V1,
    )

    with logfire.span("generate_dataset", items_per_category=items_per_category):
        tasks = build_generation_plan(items_per_category)
        results: list[GeneratedRecord] = []
        failed_count = 0

        with make_progress_bar() as progress:
            bar = progress.add_task("Generating QA items", total=len(tasks))

            for task in tasks:
                progress.update(
                    bar, description=f"{task.category} (v{task.variant + 1})"
                )
                record = _generate_one_repair_qa(task)

                if record is None:
                    failed_count += 1
                else:
                    results.append(record)

                progress.advance(bar)

        print_batch_summary(len(results), failed_count)
        return results


# --- Persistence ---


def save_qa_dataset(
    records: list[GeneratedRecord],
    path: Path = _GENERATION_OUTPUT_PATH,
) -> None:
    save_jsonl(records, path, lambda r: r.model_dump_json())
    console.print(
        f"[bold]Saved[/bold] generated {len(records)} records → [cyan]{path.relative_to(PROJECT_ROOT)}[/cyan]"
    )


# --- main ---


if __name__ == "__main__":
    records = generate_qa_dataset(items_per_category=10)
    save_qa_dataset(records)
