"""
Builds a flat generation plan from the taxonomy in schemas.py, then executes
one LLM call per task using Instructor + OpenRouter (Llama 3.1 8B). The LLM
picks a specific subcategory from a provided menu on each call. Validates output
against the RepairQA Pydantic schema and saves results to JSONL.

Run:
    uv run python -m src.generator
"""

import hashlib
import logfire
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


import os
import instructor
from dotenv import load_dotenv
from openai import OpenAI
from src.schemas import (
    CATEGORY_SUBCATEGORIES,
    GeneratedRecord,
    GenerationTask,
    RepairQA,
)


from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

MODEL_USED = "meta-llama/llama-3.1-8b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROMPT_VARIANT = "baseline_v1"
TEMPERATURE = 0.7
MAX_RETRIES = 2

_DEFAULT_OUTPUT_PATH = "data/batch_v1.jsonl"
_CACHE_DIR = Path("data/cache")

load_dotenv()


_openai_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
)

# Wrap it with Instructor for structured output
_client = instructor.from_openai(_openai_client)

logfire.configure(
    service_name="diy-pipeline",
    service_version="0.1.0",
    send_to_logfire="if-token-present",
    console=False,
)
logfire.instrument_openai(_openai_client)


# For commandline ui
console = Console()


_SYSTEM_PROMPT = """You are a home repair expert generating training data for a DIY repair assistant.

Generate one realistic DIY repair Q&A item for the given repair category.

Pick ONE subcategory from the provided list to focus on, and report your choice
in the chosen_subcategory field.

The safety_info field must name the SPECIFIC hazard and the SPECIFIC precaution for this
exact repair — generic phrases like 'be careful' or 'stay safe' are unacceptable.

Tips must be non-obvious advice a beginner would not know — not a restatement of a step."""


def _build_user_prompt(
    category: str,
    subcategory_options: list[str],
    variant_hint: str,
) -> str:
    options = ", ".join(subcategory_options)
    return (
        f"Category: {category}\n"
        f"Choose one subcategory from this list to focus on: {options}\n"
        f"{variant_hint}\n\n"
        f"Generate a complete, realistic DIY repair Q&A item."
    )


def generate_repair_qa(
    category: str,
    subcategory_options: list[str],
    variant_hint: str,
) -> RepairQA | None:
    """
    Generate one validated RepairQA item via Instructor + OpenRouter.
    Instructor enforces the RepairQA schema and auto-retries on validation failure.
    """
    try:
        return _client.chat.completions.create(
            model=MODEL_USED,
            response_model=RepairQA,
            max_retries=MAX_RETRIES,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        category, subcategory_options, variant_hint
                    ),
                },
            ],
        )
    except Exception as e:
        console.print(f"  [red][!] Generation failed ({category}): {e}[/red]")
        return None


# --- Caching ---


def hash_prompt(task: GenerationTask, prompt_variant: str) -> str:
    raw = f"{prompt_variant}|{task.category}|{task.variant}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path(prompt_hash: str) -> Path:
    """Where a cached response for this prompt hash lives on disk."""
    return _CACHE_DIR / f"{prompt_hash}.json"


def load_from_cache(prompt_hash: str) -> RepairQA | None:
    path = cache_path(prompt_hash)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return RepairQA.model_validate(data)
    except Exception:
        return None


def save_to_cache(prompt_hash: str, item: RepairQA) -> None:
    """Save a generated item to disk so future runs can skip the LLM call."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(prompt_hash).write_text(item.model_dump_json(indent=2))


# --- Planning ---


def build_variant_hint(variant: int) -> str:
    """Encourage the LLM to pick differently across variants for the same category."""
    if variant == 0:
        return "This is the first item for this category."
    return (
        f"This is item #{variant + 1} for this category. "
        f"Pick a DIFFERENT subcategory or specific problem than a typical first choice. "
        f"Vary the equipment and scenario."
    )


# --- Planning ---


def build_generation_plan(items_per_combo: int = 12) -> list[GenerationTask]:
    """
    Flatten category × variant into a list of tasks.
    With items_per_combo=12: 5 categories × 12 variants = 60 tasks total.
    """
    tasks: list[GenerationTask] = []

    for category in CATEGORY_SUBCATEGORIES.keys():
        for variant in range(items_per_combo):
            tasks.append(
                GenerationTask(
                    category=category,
                    variant=variant,
                )
            )

    return tasks


# --- Execution ---


def generate_one(task: GenerationTask) -> GeneratedRecord | None:
    """Generate one item from a planned task. Uses disk cache to skip repeat LLM calls."""
    with logfire.span(
        "generate_one",
        category=task.category,
        variant=task.variant,
    ):
        prompt_hash = hash_prompt(task, PROMPT_VARIANT)

        item = load_from_cache(prompt_hash)
        if item is None:
            item = generate_repair_qa(
                category=task.category,
                subcategory_options=CATEGORY_SUBCATEGORIES[task.category],
                variant_hint=build_variant_hint(task.variant),
            )
            if item is None:
                return None
            save_to_cache(prompt_hash, item)

        return GeneratedRecord(
            trace_id=f"qa_{uuid.uuid4().hex[:8]}",
            category=task.category,
            subcategory=item.chosen_subcategory,
            prompt_variant=PROMPT_VARIANT,
            prompt_hash=prompt_hash,
            model_used=MODEL_USED,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            record=item,
        )


def generate_dataset(items_per_combo: int = 12) -> list[GeneratedRecord]:
    """Build the plan, then execute it. Returns validated records."""
    logfire.info(
        "Starting baseline generation run",
        items_per_combo=items_per_combo,
        prompt_variant=PROMPT_VARIANT,
        model=MODEL_USED,
    )

    with logfire.span("generate_dataset", items_per_combo=items_per_combo):
        tasks = build_generation_plan(items_per_combo=items_per_combo)
        results: list[GeneratedRecord] = []
        failed_count = 0

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            bar = progress.add_task("Generating QA items", total=len(tasks))
            for task in tasks:
                progress.update(
                    bar, description=f"{task.category} (v{task.variant + 1})"
                )
                record = generate_one(task)  # ← no generator arg
                if record is None:
                    failed_count += 1
                else:
                    results.append(record)
                progress.advance(bar)

        console.print(
            f"\n[bold green]✓ Batch complete:[/bold green] "
            f"{len(results)} generated, [red]{failed_count} failed[/red]"
        )
        return results


# --- Persistence ---


def save_dataset(
    records: list[GeneratedRecord],
    path: str = _DEFAULT_OUTPUT_PATH,
) -> None:
    Path("data").mkdir(exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    console.print(f"[bold]Saved[/bold] {len(records)} records → [cyan]{path}[/cyan]")


if __name__ == "__main__":
    # Quick test — generate one item to verify Instructor + OpenRouter works
    test_item = generate_repair_qa(
        category="Plumbing Repair",
        subcategory_options=CATEGORY_SUBCATEGORIES["Plumbing Repair"],
        variant_hint="This is the first item for this category.",
    )
    if test_item:
        console.print("[bold green]✓ Test generation succeeded[/bold green]")
        console.print(f"Question: {test_item.question}")
        console.print(f"Safety: {test_item.safety_info}")
    else:
        console.print("[red]Test generation failed[/red]")

    # Full run — uncomment when the test passes
    # records = generate_dataset(items_per_combo=12)
    # save_dataset(records)
    # console.print(f"Total generated: {len(records)}")
