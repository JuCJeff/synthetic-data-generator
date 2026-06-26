"""
Shared pipeline utilities.
"""

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import logfire

from src.schemas import QARecord


def generate_uuid():
    return uuid.uuid4().hex[:8]


def save_jsonl(
    records: list[Any], path: Path, serializer: Callable[[Any], str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        for record in records:
            output_file.write(serializer(record) + "\n")


def load_labeled_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["trace_id"])
                except Exception:
                    pass
    return ids


def append_jsonl(record: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def load_qa_records(path: Path) -> tuple[list[QARecord], int]:
    """Parse a JSONL file of QARecords from any pipeline stage.

    Returns (records, parse_failure_count). Failures are structural — bad
    schema or corrupt JSON.
    """
    if not path.exists():
        return [], 0
    records = []
    failures = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(QARecord.model_validate_json(line))
                except Exception as e:
                    failures += 1
                    logfire.warning("Structural parse failure", error=str(e))
    return records, failures


def sample_balanced(
    records: list, n: int, key: Callable = lambda r: r.category
) -> list:
    """Round-robin across categories so all are represented."""
    buckets: dict = defaultdict(list)
    for r in records:
        buckets[key(r)].append(r)
    result = []
    while len(result) < n and any(buckets.values()):
        for items in list(buckets.values()):
            if items and len(result) < n:
                result.append(items.pop(0))
    return result


def print_root_cause(error: Exception) -> BaseException:
    """Unwrap exception chains including tenacity's RetryError.last_attempt."""
    cause: BaseException = error.__cause__ or error
    last_attempt = getattr(cause, "last_attempt", None)
    if last_attempt is not None:
        try:
            inner = last_attempt.exception()
            if inner is not None:
                return inner
        except Exception:
            pass
    return cause
