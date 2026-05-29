from pathlib import Path

# --- File System Setup
PROJECT_ROOT = Path(__file__).parent.parent

# --- Models ---

GENERATION_MODEL_V1 = "meta-llama/llama-3.1-8b-instruct"
GENERATION_TEMPERATURE_V1 = 0.7
MAX_RETRIES_V1 = 2

# --- Directory paths ---

GENERATED_OUTPUTS_PATH = Path("data/generated/batch_v1.jsonl")

# --- LLM Setups ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT_V1 = """You are a home repair expert generating training data for a DIY repair assistant.

Generate one realistic DIY repair Q&A item for the given repair category.

Pick ONE subcategory from the provided list to focus on, and report your choice
in the chosen_subcategory field.

The safety_info field must name the SPECIFIC hazard and the SPECIFIC precaution for this
exact repair — generic phrases like 'be careful' or 'stay safe' are unacceptable.

Tips must be non-obvious advice a beginner would not know — not a restatement of a step."""
