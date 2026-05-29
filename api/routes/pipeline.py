import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.config import GENERATED_OUTPUTS_PATH
from src.generator import generate_dataset, save_dataset
from src.validator import (
    REJECTED_OUTPUT_PATH,
    VALIDATED_OUTPUT_PATH,
    VALIDATION_REPORT_PATH,
    load_generated,
    run_quality_gate,
    save_rejected,
    save_validated,
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
        records = generate_dataset(items_per_category=items_per_category)
        save_dataset(records)
        _jobs[job_id].update({"status": "done", "count": len(records)})
    except Exception as error:
        _jobs[job_id].update({"status": "error", "error": str(error)})


@router.get("/status")
def get_status():
    return {
        "generated_exists": GENERATED_OUTPUTS_PATH.exists(),
        "validated_exists": VALIDATED_OUTPUT_PATH.exists(),
    }


@router.get("/step/generation")
def get_generation():
    records, _ = load_generated(GENERATED_OUTPUTS_PATH) if GENERATED_OUTPUTS_PATH.exists() else ([], 0)
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
        load_generated(VALIDATED_OUTPUT_PATH)
        if VALIDATED_OUTPUT_PATH.exists()
        else ([], 0)
    )
    rejected = _load_rejected(REJECTED_OUTPUT_PATH)
    return {
        "report": report,
        "validated": [r.model_dump() for r in validated],
        "rejected": rejected,
    }


@router.post("/step/validation/run")
def run_validation():
    if not GENERATED_OUTPUTS_PATH.exists():
        raise HTTPException(
            status_code=400, detail="No generated data found. Run the generation step first."
        )
    records, parse_failures = load_generated(GENERATED_OUTPUTS_PATH)
    passing, rejected_records, report = run_quality_gate(
        records, structural_failures=parse_failures
    )
    save_validated(passing, VALIDATED_OUTPUT_PATH)
    save_rejected(rejected_records, REJECTED_OUTPUT_PATH)
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report.summary(), indent=2))
    return report.summary()
