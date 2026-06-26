"""
Human labeling CLI.

Walks through validated records and collects binary pass/fail labels
across 6 quality dimensions.  Appends one lablled record to JSONL after
each completed item so progress survives an early exit.

Resumable: already-labeled trace_ids are skipped automatically.

Usage:
    uv run python -m src.labeler           # label up to 20 items
    uv run python -m src.labeler --n 30    # label up to 30 items
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.config import HUMAN_LABELER_REPORT_PATH, PROJECT_ROOT, VALIDATED_OUTPUTS_PATH
from src.schemas import EVALUATION_DIMENSIONS, EvaluationResults, LabelledRecord
from src.ui import (
    ask_dimension,
    console,
    print_dimension_result,
    print_label_item,
    print_label_saved,
    print_label_session_complete,
    print_label_session_header,
)
from src.util import append_jsonl, load_labeled_ids, load_qa_records


# --- Helpers ---
def sample_balanced(records: list, n: int) -> list:
    """Round-robin across categories so all 5 are represented"""
    buckets: dict = defaultdict(list)
    for r in records:
        buckets[r.category].append(r)
    result = []
    while len(result) < n and any(buckets.values()):
        for items in list(buckets.values()):
            if items and len(result) < n:
                result.append(items.pop(0))
    return result


def overwrite_label(label: LabelledRecord, path: Path) -> None:
    existing_lines: list[str] = []
    if path.exists():
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                    if rec.get("trace_id") != label.trace_id:
                        existing_lines.append(stripped)
                except Exception:
                    existing_lines.append(stripped)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in existing_lines:
            f.write(line + "\n")
        f.write(label.model_dump_json() + "\n")


# --- Main ---


def label_items(n: int = 20) -> None:
    records, _ = load_qa_records(VALIDATED_OUTPUTS_PATH)

    # Prompt user to finish the missing validation step
    if not records:
        console.print(
            f"No validated records found at [cyan]{VALIDATED_OUTPUTS_PATH.relative_to(PROJECT_ROOT)}[/cyan]"
        )
        console.print(
            "Run the validator first: [bold cyan]uv run python -m src.validator[/bold cyan]"
        )
        sys.exit(1)

    already_labeled = load_labeled_ids(HUMAN_LABELER_REPORT_PATH)
    queue = [r for r in records if r.trace_id not in already_labeled]

    if not queue:
        console.print(
            f"[success]All {len(records)} validated items are already labeled.[/success]"
        )
        return

    to_label = sample_balanced(queue, n)
    print_label_session_header(len(records), len(already_labeled), len(to_label))

    labeled_count = 0

    for idx, qa_record in enumerate(to_label, 1):
        print_label_item(qa_record, idx, len(to_label))

        dimension_scores: dict[str, int] = {}
        for dim_idx, dimension in enumerate(EVALUATION_DIMENSIONS, 1):
            dimension_field = dimension["field"]
            dimension_label = f"D{dim_idx}  {dimension['dimension']}"
            score = ask_dimension(
                dimension_label,
                dimension["requirement"],
                dim_idx,
                len(EVALUATION_DIMENSIONS),
            )
            if score is None:
                console.print("[warning]Item skipped.[/warning]\n")
                break
            dimension_scores[dimension_field] = score
            print_dimension_result(score)
        else:
            human_label = LabelledRecord(
                labeler="human",
                trace_id=qa_record.trace_id,
                results=EvaluationResults(**dimension_scores),
            )
            append_jsonl(human_label, HUMAN_LABELER_REPORT_PATH)
            labeled_count += 1
            dimension_fields = [d["field"] for d in EVALUATION_DIMENSIONS]
            print_label_saved(
                dimension_scores, human_label.overall_pass, dimension_fields
            )

    print_label_session_complete(
        labeled_count, len(already_labeled) + labeled_count, HUMAN_LABELER_REPORT_PATH
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human labeler — collects pass/fail on 6 quality dimensions per item"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        metavar="N",
        help="Max items to label this session (default: 20)",
    )
    args = parser.parse_args()
    label_items(n=args.n)
