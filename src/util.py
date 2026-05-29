"""
Shared pipeline utilities.
"""

from pathlib import Path
from typing import Any, Callable


def save_jsonl(
    records: list[Any], path: Path, serializer: Callable[[Any], str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        for record in records:
            output_file.write(serializer(record) + "\n")


def root_cause(error: Exception) -> BaseException:
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
