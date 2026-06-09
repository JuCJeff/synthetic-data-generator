# Closed-Loop Synthetic Data Pipeline

Generate synthetic Q&A training data, evaluate quality with LLM-as-Judge, analyze failure patterns, fix templates upstream, re-evaluate. Applied to Home DIY Repair.

## Run the App

- FastAPI backend: `uv run uvicorn api.main:app --reload`
- Streamlit frontend: `uv run streamlit run ui/streamlit_app.py`

## Sources

Usage tracking:

[- DeepSeek](https://huggingface.co/datasets/dipenbhuva/home-diy-repair-qa/viewer/default/train?row=4)
[- Logfire](https://logfire-us.pydantic.dev/juc-jeff/synthetic-data-generation)
[- Open Router](https://openrouter.ai/activity)

- Terminal Rich Color palette: <https://www.color-hex.com/color-palette/23357>
- White Beige Color palette: <https://www.color-hex.com/color-palette/1044169>
- Streamlit Color palette: <https://www.media.io/color-palette/dark-cyan-color-palette.html>
