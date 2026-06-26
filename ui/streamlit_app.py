"""
Streamlit frontend for the DIY Repair synthetic data pipeline.

Connects to the FastAPI backend at API_BASE_URL. Start the backend first:
    uv run uvicorn api.main:app --reload

Then run this app:
    uv run streamlit run streamlit_app.py
"""

import time
from collections import Counter

import requests
import streamlit as st

from src.schemas import EVALUATION_DIMENSIONS

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Home DIY Data Pipeline",
    page_icon="🔧",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = [
    "Appliance Repair",
    "Plumbing Repair",
    "Electrical Repair",
    "HVAC Maintenance",
    "General Home Repair",
]

LABEL_TARGET = 20

DIMENSIONS = [
    (
        ed["dimension"].lower().replace(" ", "_"),
        f"D{i + 1} — {ed['dimension']}",
        ed["requirement"],
    )
    for i, ed in enumerate(EVALUATION_DIMENSIONS)
]

# ── Sidebar ───────────────────────────────────────────────────────────────────

PAGES = [
    "Dashboard",
    "Step 1 - Generate",
    "Step 2 - Validate",
    "Step 3 - Human Label",
    "Step 4 - LLM Judge",
    "Step 5 - Analysis",
]

with st.sidebar:
    st.title("🔧 DIY Pipeline")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    try:
        status = requests.get(f"{API_BASE_URL}/status", timeout=2).json()
        st.caption("Pipeline Status")
        st.write("🟢 Generated data" if status.get("generated_exists") else "🟡 No generated data")
        st.write("🟢 Validated data" if status.get("validated_exists") else "🟡 No validated data")
    except Exception:
        st.error("Backend offline — start the FastAPI server first.")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get(endpoint: str, timeout: int = 5) -> dict | None:
    try:
        return requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout).json()
    except Exception:
        return None


def field(label: str, value: str) -> None:
    st.caption(label)
    st.markdown(value)


def render_record_list(records: list[dict], filter_key: str) -> None:
    category_filter = st.selectbox(
        "Filter by category", ["All"] + CATEGORIES, key=f"cat_filter_{filter_key}"
    )
    filtered = (
        records
        if category_filter == "All"
        else [r for r in records if r["category"] == category_filter]
    )
    st.markdown(f"**Showing {len(filtered)} of {len(records)} records**")
    for rec in filtered:
        r = rec["record"]
        with st.expander(f"{rec['category']}  |  `{rec['trace_id'][:8]}`"):
            field("Question", r["question"])
            field("Equipment Problem", r["equipment_problem"])
            field("Answer", r["answer"])
            field("Tools Required", " ".join(f"`{t}`" for t in r["tools_required"]))
            field("Steps", "\n".join(f"{i}. {s}" for i, s in enumerate(r["steps"], 1)))
            field("Safety Info", r["safety_info"])
            field("Tips", "\n".join(f"- {t}" for t in r["tips"]))


def poll_job(job_id: str, placeholder, status_prefix: str = "/step/generation/status") -> dict:
    while True:
        job = requests.get(f"{API_BASE_URL}{status_prefix}/{job_id}", timeout=5).json()
        status = job.get("status")
        if status == "running":
            placeholder.info("Running… this may take a minute.")
        elif status == "done":
            return job
        elif status == "error":
            placeholder.error(f"Error: {job.get('error')}")
            return job
        time.sleep(2)


# ── Pages ─────────────────────────────────────────────────────────────────────


def render_dashboard() -> None:
    st.title("Pipeline Dashboard")

    gen_data = _get("/step/generation")
    val_data = _get("/step/validation")
    label_data = _get("/step/labeling")
    judge_data = _get("/step/judge")

    steps: list[tuple[str, list[str]]] = []

    # Step 1 — Generate
    records = (gen_data or {}).get("records", [])
    if records:
        categories = Counter(r["category"] for r in records)
        categories_summary = "  |  ".join(f"{k}: {v}" for k, v in sorted(categories.items()))
        steps.append(("Step 1 — Generate", [f"{len(records)} items generated", categories_summary]))
    else:
        steps.append(("Step 1 — Generate", ["No data — run Step 1 first"]))

    # Step 2 — Validate
    report = (val_data or {}).get("report")
    if report:
        total = report.get("total_records", 0)
        valid = report.get("valid_count", 0)
        rejected = report.get("rejected_count", 0)
        pct = f"{valid / total * 100:.0f}%" if total else "—"
        steps.append(("Step 2 — Validate", [
            f"{valid}/{total} validated ({pct} pass rate)",
            f"{rejected} rejected",
        ]))
    else:
        steps.append(("Step 2 — Validate", ["No data — run Step 2 first"]))

    # Step 3 — Human Label
    if label_data:
        labeled = label_data.get("labeled", 0)
        remaining = max(0, LABEL_TARGET - labeled)
        steps.append(("Step 3 — Human Label", [
            f"{labeled}/{LABEL_TARGET} items labeled",
            "Complete ✅" if labeled >= LABEL_TARGET else f"{remaining} remaining",
        ]))
    else:
        steps.append(("Step 3 — Human Label", ["No data — run Step 3 first"]))

    # Step 4 — LLM Judge
    if judge_data:
        judged = judge_data.get("judged", 0)
        total = judge_data.get("total", 0)
        steps.append(("Step 4 — LLM Judge", [
            f"{judged}/{total} items judged",
            "Complete ✅" if judged >= total > 0 else f"{total - judged} pending",
        ]))
    else:
        steps.append(("Step 4 — LLM Judge", ["Not yet run"]))

    # Step 5 — Analysis
    steps.append(("Step 5 — Analysis", ["Requires Steps 3 and 4 — coming soon"]))

    for title, vitals in steps:
        st.subheader(title)
        for v in vitals:
            st.write(f"- {v}")
        st.divider()


def render_generate() -> None:
    st.title("Step 1 — Generate")
    st.caption("Generates DIY repair Q&A items using `meta-llama/llama-3.1-8b-instruct` via OpenRouter.")

    with st.form("generate_form"):
        n = st.slider("Items per category", min_value=1, max_value=20, value=10)
        submitted = st.form_submit_button("Run generation")

    status_slot = st.empty()
    if submitted:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/step/generation/run",
                json={"items_per_category": n},
                timeout=10,
            )
            job_id = resp.json().get("job_id")
            job = poll_job(job_id, status_slot)
            status_slot.success(f"Done — generated {job.get('count', 0)} records.")
            st.rerun()
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    st.divider()
    data = _get("/step/generation")
    records = (data or {}).get("records", [])
    if records:
        render_record_list(records, filter_key="step1")
    else:
        st.info("No generated data yet. Run the generation above.")


def render_validate() -> None:
    st.title("Step 2 — Validate")
    st.caption("Runs the quality gate: structural checks, heuristic pre-checks, dedup, and category distribution.")

    if st.button("Run validation"):
        try:
            resp = requests.post(f"{API_BASE_URL}/step/validation/run", timeout=30)
            if resp.ok:
                st.success("Validation complete.")
                st.json(resp.json())
            else:
                st.error(resp.json().get("detail", "Validation failed."))
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    st.divider()
    data = _get("/step/validation")
    if not data:
        st.warning("Could not load validation data.")
        return

    report = data.get("report")
    validated = data.get("validated", [])
    rejected = data.get("rejected", [])

    if report:
        cols = st.columns(3)
        cols[0].metric("Total generated", report.get("total_records", "—"))
        cols[1].metric("Validated", report.get("valid_count", "—"))
        cols[2].metric("Rejected", report.get("rejected_count", "—"))

    if validated:
        st.subheader(f"Validated ({len(validated)})")
        render_record_list(validated, filter_key="step2")

    if rejected:
        st.subheader(f"Rejected ({len(rejected)})")
        for rec in rejected:
            with st.expander(
                f"`{rec.get('trace_id', '?')}` — {rec.get('rejection_reason', '?')}: {rec.get('errors')[0]}"
            ):
                st.json(rec)

    if not report and not validated:
        st.info("No validation results yet. Run validation above.")


def render_human_label() -> None:
    st.title("Step 3 — Human Label")
    st.caption(f"Label up to {LABEL_TARGET} validated items across 6 quality dimensions.")

    try:
        label_data = requests.get(f"{API_BASE_URL}/step/labeling", timeout=5).json()
        labels_list = label_data.get("labels", [])
        labels_by_id = {r["trace_id"]: r for r in labels_list}
        labeled_ids = set(labels_by_id.keys())
        total_labeled = label_data.get("labeled", 0)
    except Exception:
        labels_list, labels_by_id, labeled_ids = [], {}, set()
        total_labeled = 0
        st.error("Could not load labeling state from backend.")

    try:
        all_items = requests.get(f"{API_BASE_URL}/step/validation", timeout=5).json().get("validated", [])
    except Exception:
        all_items = []
        st.error("Could not load validated items.")

    if not all_items:
        st.info("No validated items. Complete Steps 1 and 2 first.")
        st.stop()

    queue = [item for item in all_items if item["trace_id"] not in labeled_ids][:LABEL_TARGET]

    st.progress(
        min(total_labeled, LABEL_TARGET) / LABEL_TARGET,
        text=f"Labeled {min(total_labeled, LABEL_TARGET)} of {LABEL_TARGET} items",
    )

    if not queue:
        st.success(f"{LABEL_TARGET} items labeled! Proceed to Step 4.")
        st.stop()

    relabel_trace_id = st.session_state.get("relabel_trace_id")
    if relabel_trace_id:
        item = next((i for i in all_items if i["trace_id"] == relabel_trace_id), None)
        if item is None:
            st.session_state.pop("relabel_trace_id", None)
            st.rerun()
        is_relabel = True
    else:
        idx = st.session_state.get("label_idx", 0)
        idx = min(idx, len(queue) - 1)
        item = queue[idx]
        is_relabel = False

    r = item.get("record", {})

    if is_relabel and relabel_trace_id in labels_by_id:
        saved = labels_by_id[relabel_trace_id]
        for dim_key, _, _ in DIMENSIONS:
            widget_key = f"{dim_key}_{item['trace_id']}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = saved.get(dim_key, 1)

    col_record, col_labels = st.columns([3, 2])

    with col_record:
        relabel_tag = "  *(relabeling)*" if is_relabel else ""
        st.caption(
            f"{item.get('category')}  |  `{item['trace_id'][:8]}`  |  {r.get('chosen_subcategory', '—')}{relabel_tag}"
        )
        with st.expander("Question", expanded=True):
            st.write(r.get("question", "—"))
        with st.expander("Answer", expanded=True):
            st.write(r.get("answer", "—"))
        with st.expander("Equipment Problem"):
            st.write(r.get("equipment_problem", "—"))
        with st.expander("Tools Required"):
            st.markdown(" ".join(f"`{t}`" for t in r.get("tools_required", [])))
        with st.expander("Steps"):
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(r.get("steps", []), 1)))
        with st.expander("Safety Info"):
            st.write(r.get("safety_info", "—"))
        with st.expander("Tips"):
            st.markdown("\n".join(f"- {t}" for t in r.get("tips", [])))

    with col_labels:
        with st.form(f"label_form_{item['trace_id']}"):
            st.caption("YOUR LABELS")
            dimension_labels = {}
            for dim_key, dim_label, dim_hint in DIMENSIONS:
                dimension_labels[dim_key] = st.radio(
                    f"**{dim_label}**",
                    options=[1, 0],
                    format_func=lambda v: "Pass ✅" if v == 1 else "Fail ❌",
                    horizontal=True,
                    help=dim_hint,
                    key=f"{dim_key}_{item['trace_id']}",
                )
                st.divider()

            prev_col, next_col = st.columns(2)
            history_ids = [r["trace_id"] for r in labels_list]
            can_go_back = is_relabel or (not is_relabel and idx > 0) or bool(history_ids)

            with prev_col:
                if can_go_back and st.form_submit_button("← Previous", use_container_width=True):
                    if is_relabel:
                        current_pos = history_ids.index(relabel_trace_id) if relabel_trace_id in history_ids else -1
                        if current_pos > 0:
                            prev_id = history_ids[current_pos - 1]
                            for dim_key, _, _ in DIMENSIONS:
                                st.session_state.pop(f"{dim_key}_{prev_id}", None)
                            st.session_state["relabel_trace_id"] = prev_id
                        else:
                            st.session_state.pop("relabel_trace_id", None)
                    elif idx > 0:
                        st.session_state["label_idx"] = idx - 1
                    elif history_ids:
                        last_id = history_ids[-1]
                        for dim_key, _, _ in DIMENSIONS:
                            st.session_state.pop(f"{dim_key}_{last_id}", None)
                        st.session_state["relabel_trace_id"] = last_id
                    st.rerun()

            with next_col:
                submit_text = "Update Label →" if is_relabel else "Submit Label →"
                if st.form_submit_button(submit_text, type="primary", use_container_width=True):
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/step/labeling/submit",
                            json={"trace_id": item["trace_id"], **dimension_labels, "relabel": is_relabel},
                            timeout=10,
                        )
                        if resp.ok:
                            if is_relabel:
                                st.session_state.pop("relabel_trace_id", None)
                            else:
                                st.session_state["label_idx"] = idx
                            st.rerun()
                        else:
                            st.error(resp.json().get("detail", "Submission failed."))
                    except Exception as exc:
                        st.error(f"Request failed: {exc}")


def render_llm_judge() -> None:
    st.title("Step 4 — LLM Judge")
    st.caption("Scores all validated items on 6 quality dimensions using `deepseek-v4-flash`.")

    data = _get("/step/judge")
    if data:
        total = data.get("total", 0)
        judged = data.get("judged", 0)
        pending = data.get("pending", 0)
        cols = st.columns(3)
        cols[0].metric("Total validated", total)
        cols[1].metric("Judged", judged)
        cols[2].metric("Pending", pending)

    if st.button("Run LLM Judge", disabled=not data or data.get("total", 0) == 0):
        try:
            resp = requests.post(f"{API_BASE_URL}/step/judge/run", timeout=10)
            job_id = resp.json().get("job_id")
            status_slot = st.empty()
            job = poll_job(job_id, status_slot, status_prefix="/step/judge/status")
            judged = job.get("judged", 0)
            total = job.get("total", 0)
            status_slot.success(f"Done — judged {judged}/{total} items.")
            st.rerun()
        except Exception as exc:
            st.error(f"Request failed: {exc}")


def render_analysis() -> None:
    st.title("Step 5 — Analysis")
    st.info("Analysis and visualisation coming soon. Charts will appear here once Steps 3 and 4 are complete.")


# ── Dispatch ──────────────────────────────────────────────────────────────────

{
    "Dashboard": render_dashboard,
    "Step 1 - Generate": render_generate,
    "Step 2 - Validate": render_validate,
    "Step 3 - Human Label": render_human_label,
    "Step 4 - LLM Judge": render_llm_judge,
    "Step 5 - Analysis": render_analysis,
}[page]()
