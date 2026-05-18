"""
Builds a flat generation plan from the taxonomy in schemas.py, then executes
one LLM call per task using DSPy + a chosen LLM model. The LLM picks a
specific subcategory from a provided menu on each call, encouraging variety
while keeping the number of LLM calls low. Validates output against the
RepairQA Pydantic schema and saves results to JSONL.

Run:
    uv run python -m generator
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


import dspy
from dotenv import load_dotenv
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


from src.schemas import (
    CATEGORY_SUBCATEGORIES,
    DIFFICULTIES,
    GeneratedRecord,
    GenerationTask,
    RepairQA,
)

MODEL_PROVIDER = "openai"
MODEL_USED = "gpt-4o-mini"
PROMPT_VARIANT = "baseline_v1"

_DEFAULT_OUTPUT_PATH = "data/batch_v1.jsonl"
_CACHE_DIR = Path("data/cache")

load_dotenv()
dspy.configure(lm=dspy.LM(model=f"{MODEL_PROVIDER}/{MODEL_USED}", temperature=0.8))

# For commandline ui
console = Console()


class RepairQASignature(dspy.Signature):
    """
    You are a home repair expert generating training data for a DIY repair assistant.

    Generate one realistic DIY repair Q&A item for the given repair category, sub category 
    and difficulty.

    Difficulty guides the complexity of the repair:
    - beginner: simple, common repairs a first-timer could handle
    - intermediate: multi-step repairs requiring some experience
    - advanced: complex but still safe for a capable DIYer — no gas lines or panel work

    The safety_info field must name the SPECIFIC hazard and the SPECIFIC precaution
    for this exact repair — generic phrases like 'be careful' or 'stay safe' will fail.

    Tips must be non-obvious advice a beginner would not know — not a restatement of a step.
    """

    category: str = dspy.InputField(
        description="The repair category to generate a Q&A item for"
    )
    subcategory_options: list[str] = dspy.InputField(
        description="The available equipment or repair types to choose from for this category"
    )
    difficulty: str = dspy.InputField(
        description="The difficulty tier: beginner, intermediate, or advanced"
    )
    variant_hint: str = dspy.InputField(
        description="A hint encouraging variety on repeated calls for the same category and difficulty"
    )
    question: str = dspy.OutputField(
        description="A specific, realistic question a homeowner would ask"
    )
    answer: str = dspy.OutputField(
        description=(
            "A detailed narrative answer (700-1300 characters) covering "
            "tools, steps, safety warnings, and tips"
        )
    )
    equipment_problem: str = dspy.OutputField(
        description="The specific problem being addressed, e.g. 'dripping faucet'"
    )
    chosen_subcategory: str = dspy.OutputField(
        description="Which subcategory from the provided list this item focuses on"
    )
    tools_required: list[str] = dspy.OutputField(
        description="Tools a homeowner owns or can buy at a hardware store for under $50"
    )
    steps: list[str] = dspy.OutputField(
        description="At least 3 ordered, concrete repair steps"
    )
    safety_info: str = dspy.OutputField(
        description="The SPECIFIC hazard and SPECIFIC precaution for this repair"
    )
    tips: list[str] = dspy.OutputField(
        description="At least 1 non-obvious, task-specific tip"
    )
    difficulty_out: str = dspy.OutputField(
        description="Echo back the difficulty tier: beginner, intermediate, or advanced"
    )


class RepairQAGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(RepairQASignature)

    def forward(
        self,
        category: str,
        subcategory_options: list[str],
        difficulty: str,
        variant_hint: str,
    ) -> tuple[RepairQA, str] | None:
        """Returns (RepairQA, chosen_subcategory) or None on failure."""
        try:
            result = self.generate(
                category=category,
                subcategory_options=subcategory_options,
                difficulty=difficulty,
                variant_hint=variant_hint,
            )

            # Pass DSPy output through Pydantic validation
            item = RepairQA(
                question=result.question,
                answer=result.answer,
                equipment_problem=result.equipment_problem,
                tools_required=result.tools_required,
                steps=result.steps,
                safety_info=result.safety_info,
                tips=result.tips,
                difficulty=result.difficulty_out,
            )
            return item, result.chosen_subcategory
        except Exception as e:
            console.print(
                f"  [red][!] Generation failed " f"({category}/{difficulty}): {e}[/red]"
            )
            return None


# --- Helpers ---


def hash_prompt(task: GenerationTask, prompt_variant: str) -> str:
    """Short fingerprint of the task and prompt variant — supports dedup in Step 2."""
    raw = f"{prompt_variant}|{task.category}|{task.difficulty}|{task.variant}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path(prompt_hash: str) -> Path:
    """Where a cached response for this prompt hash lives on disk."""
    return _CACHE_DIR / f"{prompt_hash}.json"


def load_from_cache(prompt_hash: str) -> tuple[RepairQA, str] | None:
    """Load a cached (RepairQA, chosen_subcategory) tuple, or None if not cached."""
    path = cache_path(prompt_hash)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        item = RepairQA.model_validate(data["record"])
        return item, data["chosen_subcategory"]
    except Exception:
        # Cache file is stale or corrupt — ignore and regenerate
        return None


def save_to_cache(prompt_hash: str, item: RepairQA, chosen_subcategory: str) -> None:
    """Save a generated item to disk so future runs can skip the LLM call."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "record": item.model_dump(),
        "chosen_subcategory": chosen_subcategory,
    }
    cache_path(prompt_hash).write_text(json.dumps(payload, indent=2))


def build_variant_hint(variant: int) -> str:
    """Encourage the LLM to pick differently across variants for the same category."""
    if variant == 0:
        return "This is the first item for this category and difficulty."
    return (
        f"This is item #{variant + 1} for this category and difficulty. "
        f"Pick a DIFFERENT subcategory or specific problem than a typical first choice. "
        f"Vary the equipment and scenario."
    )


# --- Planning ---


def build_generation_plan(items_per_combo: int = 4) -> list[GenerationTask]:
    """
    Flatten category × difficulty × variant into a list of tasks.
    The LLM picks the subcategory at generation time from the full menu.

    With items_per_combo=4: 5 categories × 3 difficulties × 4 variants = 60 tasks total.
    """
    tasks: list[GenerationTask] = []

    for category in CATEGORY_SUBCATEGORIES.keys():
        for difficulty in DIFFICULTIES:
            for variant in range(items_per_combo):
                tasks.append(
                    GenerationTask(
                        category=category,
                        subcategory="",  # filled in by the LLM at generation time
                        difficulty=difficulty,
                        variant=variant,
                    )
                )

    return tasks


# --- Execution ---


def generate_one(
    generator: RepairQAGenerator,
    task: GenerationTask,
) -> GeneratedRecord | None:
    """Generate one item from a planned task. Uses disk cache to skip repeat LLM calls."""
    prompt_hash = hash_prompt(task, PROMPT_VARIANT)
    subcategory_options = CATEGORY_SUBCATEGORIES[task.category]
    variant_hint = build_variant_hint(task.variant)

    # Cache check — skip the LLM entirely if we've generated this exact task before
    cached = load_from_cache(prompt_hash)
    if cached is not None:
        item, chosen_subcategory = cached
    else:
        result = cast(
            tuple[RepairQA, str] | None,
            generator(
                category=task.category,
                subcategory_options=subcategory_options,
                difficulty=task.difficulty,
                variant_hint=variant_hint,
            ),
        )
        if result is None:
            return None
        item, chosen_subcategory = result
        save_to_cache(prompt_hash, item, chosen_subcategory)

    return GeneratedRecord(
        trace_id=f"qa_{uuid.uuid4().hex[:8]}",
        category=task.category,
        subcategory=chosen_subcategory,
        difficulty=task.difficulty,
        prompt_variant=PROMPT_VARIANT,
        prompt_hash=prompt_hash,
        model_used=MODEL_USED,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        record=item,
    )


def generate_dataset(items_per_combo: int = 4) -> list[GeneratedRecord]:
    """Build the plan, then execute it. Returns validated records."""
    tasks = build_generation_plan(items_per_combo=items_per_combo)
    generator = RepairQAGenerator()
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
                bar,
                description=f"{task.category} / {task.difficulty} (v{task.variant + 1})",
            )

            record = generate_one(generator, task)
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
    records = generate_dataset(items_per_combo=4)
    save_dataset(records)
    console.print(f"Total generated: {len(records)}")
