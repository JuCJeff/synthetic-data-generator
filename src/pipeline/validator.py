"""
Quality gate for generated DIY repair Q&A items.

Two-stage filter (fast → cheap → batch):
  1. Per-dim heuristic pre-checks — drop obviously bad content
  2. Dedup + category distribution — flag batches that need regeneration

Structural validation happens at load time inside load_validated(). Parse
failures are counted there and surfaced in the report via structural_failure_count.
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import logfire
from rapidfuzz import fuzz

from src.config import (
    GENERATED_OUTPUTS_PATH,
    VALIDATED_OUTPUTS_PATH,
    REJECTED_OUTPUTS_PATH,
    VALIDATION_REPORT_PATH,
)
from src.instrumentation import configure_logfire
from src.schemas import CATEGORY_SUBCATEGORIES, QARecord
from src.ui import console, print_error
from src.util import load_qa_records, save_jsonl

# --- Config ---

# Minimum safety_info length to not be considered a placeholder
_MIN_SAFETY_INFO_LENGTH = 15

# Phrases that indicate a generic, non-specific safety info
_GENERIC_SAFETY_PHRASES = [
    "be careful.",
    "stay safe.",
    "use caution.",
    "exercise caution.",
    "be cautious.",
    "take care.",
    "be safe.",
    "safety first.",
]

# Tools that are trade-only, not homeowner realistic
_PROFESSIONAL_TOOL_BLOCKLIST = [
    "pipe threading machine",
    "conduit bender",
    "arc welder",
    "plasma cutter",
    "hydraulic press",
    "oscilloscope",
    "megohmmeter",
    "commercial grade",
]

# Fuzzy similarity threshold for dedup (0-100). 90 = near-identical questions
_DEDUP_SIMILARITY_THRESHOLD = 98

# Each category must make up at least this fraction of the validated set
_MIN_CATEGORY_FRACTION = 0.15


# --- Result types ---


@dataclass
class HeuristicFailure:
    field: str
    reason: str


@dataclass
class ValidationResult:
    trace_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    failed_fields: list[str] = field(default_factory=list)


@dataclass
class RejectedRecord:
    record: QARecord
    rejection_reason: str  # "heuristic" | "duplicate"
    errors: list[str] = field(default_factory=list)
    failed_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.record.trace_id,
            "rejection_reason": self.rejection_reason,
            "errors": self.errors,
            "failed_fields": self.failed_fields,
            "record": self.record.model_dump(),
        }


@dataclass
class ValidationReport:
    total_records: int = 0
    structural_failure_count: int = 0
    valid_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    field_error_counts: Counter = field(default_factory=Counter)
    category_distribution: dict[str, int] = field(default_factory=dict)
    distribution_ok: bool = True

    @property
    def success_rate(self) -> str:
        if self.total_records == 0:
            return "No total records found"
        rate = self.valid_count / self.total_records
        return f"{rate * 100:.1f}%"

    def summary(self) -> dict:
        return {
            "total_records": self.total_records,
            "structural_failure_count": self.structural_failure_count,
            "valid_count": self.valid_count,
            "rejected_count": self.rejected_count,
            "duplicate_count": self.duplicate_count,
            "success_rate": self.success_rate,
            "field_error_frequency": dict(self.field_error_counts),
            "category_distribution": self.category_distribution,
            "distribution_ok": self.distribution_ok,
        }


# --- Heuristic pre-checks ---


def _check_safety_info(safety_info: str) -> HeuristicFailure | None:
    if len(safety_info.strip()) < _MIN_SAFETY_INFO_LENGTH:
        return HeuristicFailure(
            field="safety_info",
            reason=f"Too short ({len(safety_info)} chars) — likely a placeholder",
        )
    for phrase in _GENERIC_SAFETY_PHRASES:
        if phrase in safety_info.lower():
            return HeuristicFailure(
                field="safety_info",
                reason=f"Contains generic phrase: '{phrase}'",
            )
    return None


def _check_tools(tools: list[str]) -> HeuristicFailure | None:
    tools_text = " ".join(tools).lower()
    for blocked in _PROFESSIONAL_TOOL_BLOCKLIST:
        if blocked in tools_text:
            return HeuristicFailure(
                field="tools_required",
                reason=f"Contains trade-only tool: '{blocked}'",
            )
    return None


def check_heuristics(record: QARecord) -> ValidationResult:
    """Fast content checks — catch obviously bad items before the LLM judge."""
    qa = record.record
    failures: list[HeuristicFailure] = []

    safety_failure = _check_safety_info(qa.safety_info)
    if safety_failure:
        failures.append(safety_failure)

    tool_failure = _check_tools(qa.tools_required)
    if tool_failure:
        failures.append(tool_failure)

    if failures:
        logfire.warning(
            "Heuristic pre-check failed",
            trace_id=record.trace_id,
            failures=[f.reason for f in failures],
        )
        return ValidationResult(
            trace_id=record.trace_id,
            is_valid=False,
            errors=[f.reason for f in failures],
            failed_fields=[f.field for f in failures],
        )

    return ValidationResult(trace_id=record.trace_id, is_valid=True)


# --- Dedup ---


def deduplicate(
    records: list[QARecord],
) -> tuple[list[QARecord], list[RejectedRecord]]:
    """
    Remove near-duplicate questions using fuzzy string similarity.
    Returns (deduplicated_records, rejected_records).
    """
    seen_questions: list[str] = []
    unique: list[QARecord] = []
    rejected: list[RejectedRecord] = []

    for record in records:
        question = record.record.question
        is_duplicate = any(
            fuzz.ratio(question, seen) >= _DEDUP_SIMILARITY_THRESHOLD
            for seen in seen_questions
        )
        if is_duplicate:
            logfire.info(
                "Duplicate removed", trace_id=record.trace_id, question=question[:80]
            )
            rejected.append(
                RejectedRecord(
                    record=record,
                    rejection_reason="duplicate",
                    errors=["Near-duplicate question detected"],
                )
            )
        else:
            seen_questions.append(question)
            unique.append(record)

    return unique, rejected


# --- Category distribution ---


def check_distribution(records: list[QARecord]) -> tuple[dict[str, int], bool]:
    """
    Verify each category meets the minimum fraction of the validated set.
    Returns (counts_per_category, distribution_is_ok).
    """
    counts: Counter = Counter(r.category for r in records)
    total = len(records)
    distribution_ok = True

    for category in CATEGORY_SUBCATEGORIES:
        count = counts.get(category, 0)
        fraction = count / total if total > 0 else 0
        if fraction < _MIN_CATEGORY_FRACTION:
            logfire.warning(
                "Category under-represented",
                category=category,
                count=count,
                fraction=f"{fraction:.1%}",
                threshold=f"{_MIN_CATEGORY_FRACTION:.1%}",
            )
            distribution_ok = False

    return dict(counts), distribution_ok


# --- Orchestrator ---


def run_quality_gate(
    records: list[QARecord],
    structural_failures: int = 0,
) -> tuple[list[QARecord], list[RejectedRecord], ValidationReport]:
    """
    Run all gate stages. Returns (passing_records, rejected_records, report).

    structural_failures: parse errors counted by load_validated() — included
    in total_records so the report reflects the full picture.
    Per-item heuristic failures → drop the item.
    Batch failures (distribution) → flagged in the report for the caller to
    decide whether to regenerate.
    """
    report = ValidationReport(
        total_records=len(records) + structural_failures,
        structural_failure_count=structural_failures,
    )
    passing: list[QARecord] = []
    rejected: list[RejectedRecord] = []

    with logfire.span("quality_gate", total_input=len(records)):
        for record in records:
            heuristic = check_heuristics(record)
            if not heuristic.is_valid:
                report.rejected_count += 1
                for f in heuristic.failed_fields:
                    report.field_error_counts[f] += 1
                rejected.append(
                    RejectedRecord(
                        record=record,
                        rejection_reason="heuristic",
                        errors=heuristic.errors,
                        failed_fields=heuristic.failed_fields,
                    )
                )
                continue

            passing.append(record)

        passing, dupe_rejections = deduplicate(passing)
        rejected.extend(dupe_rejections)
        report.duplicate_count = len(dupe_rejections)

        report.valid_count = len(passing)
        report.category_distribution, report.distribution_ok = check_distribution(
            passing
        )

        logfire.info("Quality gate complete", **report.summary())

    return passing, rejected, report


# --- Persistence ---


def save_validated(records: list[QARecord], path: Path) -> None:
    save_jsonl(records, path, lambda r: r.model_dump_json())


def save_rejected(records: list[RejectedRecord], path: Path) -> None:
    save_jsonl(records, path, lambda r: json.dumps(r.to_dict()))


if __name__ == "__main__":

    configure_logfire()

    records, parse_failures = load_qa_records(GENERATED_OUTPUTS_PATH)
    if not records:
        print_error(
            "Pipeline Dependency Missing",
            f"{GENERATED_OUTPUTS_PATH} has not been generated yet.",
            suggestion="Run the generator first: uv run python -m src.generator",
        )
        sys.exit(1)

    console.print(f"Loaded {len(records)} records ({parse_failures} failed to parse).")

    passing, rejected, report = run_quality_gate(
        records, structural_failures=parse_failures
    )

    save_validated(passing, VALIDATED_OUTPUTS_PATH)
    save_rejected(rejected, REJECTED_OUTPUTS_PATH)

    report_path = VALIDATION_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.summary(), indent=2))

    console.print(
        f"Gate complete: [success]{report.valid_count} passed[/success], "
        f"[error]{report.rejected_count} rejected[/error], "
        f"{report.duplicate_count} duplicates removed"
    )
