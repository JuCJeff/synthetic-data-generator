import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.config import GENERATED_OUTPUTS_PATH
from src.pipeline.generator import generate_qa_dataset, save_qa_dataset
from src.pipeline.human_labeler import (
    HUMAN_LABELS_PATH,
    append_label,
    load_labeled_ids,
    overwrite_label,
)
from src.schemas import EvaluationResults, LabelledRecord
from src.pipeline.validator import (
    run_quality_gate,
    save_rejected,
    save_validated,
)
from src.util import load_qa_records
from src.config import (
    VALIDATED_OUTPUTS_PATH,
    REJECTED_OUTPUTS_PATH,
    VALIDATION_REPORT_PATH,
)

router = APIRouter()

_jobs: dict[str, dict] = {}


def _load_rejected(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


class GenerateRequest(BaseModel):
    items_per_category: int = 10


def _run_generation(job_id: str, items_per_category: int) -> None:
    try:
        _jobs[job_id]["status"] = "running"
        records = generate_qa_dataset(items_per_category=items_per_category)
        save_qa_dataset(records)
        _jobs[job_id].update({"status": "done", "count": len(records)})
    except Exception as error:
        _jobs[job_id].update({"status": "error", "error": str(error)})


@router.get("/status")
def get_status():
    return {
        "generated_exists": GENERATED_OUTPUTS_PATH.exists(),
        "validated_exists": VALIDATED_OUTPUTS_PATH.exists(),
    }


@router.get("/step/generation")
def get_generation():
    records, _ = (
        load_qa_records(GENERATED_OUTPUTS_PATH)
        if GENERATED_OUTPUTS_PATH.exists()
        else ([], 0)
    )
    return {"records": [r.model_dump() for r in records]}


@router.post("/step/generation/run")
def run_generation(body: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "pending", "count": 0}
    background_tasks.add_task(_run_generation, job_id, body.items_per_category)
    return {"job_id": job_id}


@router.get("/step/generation/status/{job_id}")
def get_generation_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/step/validation")
def get_validation():
    report = (
        json.loads(VALIDATION_REPORT_PATH.read_text())
        if VALIDATION_REPORT_PATH.exists()
        else None
    )
    validated, _ = (
        load_qa_records(VALIDATED_OUTPUTS_PATH)
        if VALIDATED_OUTPUTS_PATH.exists()
        else ([], 0)
    )
    rejected = _load_rejected(REJECTED_OUTPUTS_PATH)
    return {
        "report": report,
        "validated": [r.model_dump() for r in validated],
        "rejected": rejected,
    }


@router.post("/step/validation/run")
def run_validation():
    if not GENERATED_OUTPUTS_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="No generated data found. Run the generation step first.",
        )
    records, parse_failures = load_qa_records(GENERATED_OUTPUTS_PATH)
    passing, rejected_records, report = run_quality_gate(
        records, structural_failures=parse_failures
    )
    save_validated(passing, VALIDATED_OUTPUTS_PATH)
    save_rejected(rejected_records, REJECTED_OUTPUTS_PATH)
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report.summary(), indent=2))
    return report.summary()


# --- Step 3: Human Labeling ---


def _load_human_labelled_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    human_labeled_records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    human_labeled_records.append(json.loads(line))
                except Exception:
                    pass
    return human_labeled_records


class LabelSubmission(BaseModel):
    trace_id: str
    answer_completeness: int
    safety_specificity: int
    tool_realism: int
    scope_appropriateness: int
    context_clarity: int
    tip_usefulness: int
    relabel: bool = False


@router.get("/step/labeling")
def get_labeling():
    records, _ = (
        load_qa_records(VALIDATED_OUTPUTS_PATH)
        if VALIDATED_OUTPUTS_PATH.exists()
        else ([], 0)
    )
    labels = _load_human_labelled_records(HUMAN_LABELS_PATH)
    labeled_ids = {r["trace_id"] for r in labels}
    return {
        "total_validated": len(records),
        "labeled": len(labels),
        "remaining": len([r for r in records if r.trace_id not in labeled_ids]),
        "labels": labels,
    }


@router.get("/step/labeling/next")
def get_next_item():
    if not VALIDATED_OUTPUTS_PATH.exists():
        raise HTTPException(
            status_code=400, detail="No validated data found. Run validation first."
        )
    records, _ = load_qa_records(VALIDATED_OUTPUTS_PATH)
    labeled_ids = load_labeled_ids(HUMAN_LABELS_PATH)
    queue = [r for r in records if r.trace_id not in labeled_ids]
    if not queue:
        return {"item": None, "remaining": 0}
    return {"item": queue[0].model_dump(), "remaining": len(queue)}


@router.post("/step/labeling/submit")
def submit_label(body: LabelSubmission):
    if not VALIDATED_OUTPUTS_PATH.exists():
        raise HTTPException(status_code=400, detail="No validated data found.")
    records, _ = load_qa_records(VALIDATED_OUTPUTS_PATH)
    if body.trace_id not in {r.trace_id for r in records}:
        raise HTTPException(
            status_code=404,
            detail=f"trace_id '{body.trace_id}' not found in validated records.",
        )
    already_labeled = body.trace_id in load_labeled_ids(HUMAN_LABELS_PATH)
    if already_labeled and not body.relabel:
        raise HTTPException(
            status_code=409, detail=f"trace_id '{body.trace_id}' is already labeled."
        )
    score_fields = set(EvaluationResults.model_fields)
    label = LabelledRecord(
        trace_id=body.trace_id,
        labeler="human",
        results=EvaluationResults(**{f: getattr(body, f) for f in score_fields}),
    )
    if body.relabel:
        overwrite_label(label, HUMAN_LABELS_PATH)
    else:
        append_label(label, HUMAN_LABELS_PATH)
    return label.model_dump()
