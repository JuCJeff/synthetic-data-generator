from pathlib import Path

# --- File System Setup
PROJECT_ROOT = Path(__file__).parent.parent


# --- API Provider Setups ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# --- Models ---
# Synthetic Data generation model
GENERATION_MODEL_V1 = "meta-llama/llama-3.1-8b-instruct"
GENERATION_TEMPERATURE_V1 = 0.7
MAX_RETRIES_V1 = 2

# --- Directory paths ---
# Generation
GENERATED_OUTPUTS_PATH = Path("data/generated/batch_v1.jsonl")

# Validation
VALIDATED_OUTPUTS_PATH = Path("data/validated/validated_records.jsonl")
REJECTED_OUTPUTS_PATH = Path("data/validated/rejected_records.jsonl")
VALIDATION_REPORT_PATH = Path("data/validated/validation_report.json")


# --- Prompts ---
# Synthetic data generation
SYSTEM_PROMPT_V1 = """You are a home repair expert generating training data for a DIY repair assistant.

Generate one realistic DIY repair Q&A item for the given repair category.

Pick ONE subcategory from the provided list to focus on, and report your choice
in the chosen_subcategory field.

The safety_info field must name the SPECIFIC hazard and the SPECIFIC precaution for this
exact repair — generic phrases like 'be careful' or 'stay safe' are unacceptable.

Tips must be non-obvious advice a beginner would not know — not a restatement of a step."""
