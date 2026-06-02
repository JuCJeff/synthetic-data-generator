"""
Pydantic schemas used across the DIY repair pipeline.

Single source of truth for:
- Data shapes
- Generation planning
- Type aliases
"""

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "Appliance Repair",
    "Plumbing Repair",
    "Electrical Repair",
    "HVAC Maintenance",
    "General Home Repair",
]

CATEGORY_SUBCATEGORIES: dict[Category, list[str]] = {
    "Appliance Repair": [
        "refrigerators",
        "washing machines",
        "dryers",
        "dishwashers",
        "ovens",
    ],
    "Plumbing Repair": [
        "leaks",
        "clogs",
        "fixture repairs",
        "pipe problems",
    ],
    "Electrical Repair": [
        "outlet replacement",
        "switch repair",
        "light fixture installation",
    ],
    "HVAC Maintenance": [
        "filter changes",
        "thermostat issues",
        "vent cleaning",
        "basic troubleshooting",
    ],
    "General Home Repair": [
        "drywall",
        "doors and windows",
        "flooring",
        "basic carpentry",
    ],
}


class RepairQA(BaseModel):
    """A single DIY home repair Q&A training item."""

    question: str = Field(
        description="A realistic DIY repair question from a homeowner"
    )
    answer: str = Field(
        description="A clear, actionable answer with step-by-step guidance"
    )
    equipment_problem: str = Field(
        description='The specific problem being addressed (e.g. "dripping faucet")'
    )
    tools_required: list[str] = Field(
        description="Tools a typical homeowner would realistically own", min_length=1
    )
    steps: list[str] = Field(description="Ordered, numbered repair steps", min_length=3)
    safety_info: str = Field(
        description="Relevant safety warnings and precautions", min_length=1
    )
    tips: list[str] = Field(
        description="Practical tips to make the repair easier or more reliable",
        min_length=1,
    )
    chosen_subcategory: str = Field(
        description="The subcategory from the provided list that this item focuses on"
    )

    # Small models sometimes returns list fields as JSON strings — parse them back to list
    @field_validator("tools_required", "steps", "tips", mode="before")
    @classmethod
    def parse_stringified_list(cls, v: str | list) -> list:
        if isinstance(v, str):
            return json.loads(v)
        return v


class GeneratedRecord(BaseModel):
    """Wrapper that separates pipeline metadata from content"""

    trace_id: str = Field(
        description="Unique identifier for this item across all pipeline steps"
    )
    category: Category = Field(
        description="Which of the 5 repair categories this item belongs to"
    )
    subcategory: str = Field(
        description="The specific equipment or repair type within the category"
    )
    prompt_variant: str = Field(description="Which generation prompt template was used")
    prompt_hash: str = Field(
        description="Hash of the exact prompt — supports dedup in Step 2"
    )
    model_used: str = Field(description="Which LLM model produced this item")
    generation_timestamp: str = Field(
        description="UTC timestamp of when this item was generated"
    )
    record: RepairQA = Field(description="The actual Q&A content")


class LabelRecord(BaseModel):
    """Quality evaluation of a single RepairQA item across 6 dimensions."""

    trace_id: str = Field(
        description="Unique identifier linking this label to its source RepairQA item"
    )
    labeler: Literal["human", "llm_judge"] = Field(
        description="Who produced this label"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp when this label was produced",
    )

    answer_completeness: int = Field(
        description="1 if answer gives enough detail to complete the repair, 0 if it stops short",
        ge=0,
        le=1,
    )
    safety_specificity: int = Field(
        description="1 if safety_info names the specific hazard and precaution, 0 if generic",
        ge=0,
        le=1,
    )
    tool_realism: int = Field(
        description="1 if all tools are homeowner-realistic, 0 if any are trade-only",
        ge=0,
        le=1,
    )
    scope_appropriateness: int = Field(
        description="1 if repair is within DIY capability or correctly defers to a pro, 0 otherwise",
        ge=0,
        le=1,
    )
    context_clarity: int = Field(
        description="1 if question and answer have enough context to understand the problem, 0 if vague",
        ge=0,
        le=1,
    )
    tip_usefulness: int = Field(
        description="1 if tips are non-obvious and task-specific, 0 if generic",
        ge=0,
        le=1,
    )

    @property
    def overall_pass(self) -> bool:
        return all(
            [
                self.answer_completeness,
                self.safety_specificity,
                self.tool_realism,
                self.scope_appropriateness,
                self.context_clarity,
                self.tip_usefulness,
            ]
        )


class GenerationTask(BaseModel):
    """A single planned generation. Built before any LLM calls happen"""

    category: Category
    variant: int = Field(
        description="Counter for multiple items of the same combination"
    )
