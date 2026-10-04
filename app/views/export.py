import io
import json

import pandas as pd
import streamlit as st

from app import state, ui
from core import report
from core.profile import describe_step
from db.repo import save_dataset, save_pipeline
from db.session import SessionLocal


def _excel(df, steps, prof) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Cleaned data", index=False)
        pd.DataFrame({"step": range(1, len(steps) + 1), "action": [describe_step(x) for x in steps],
                      "details": [json.dumps(x, default=str) for x in steps]}).to_excel(xw, sheet_name="Cleaning log", index=False)
        pd.DataFrame([prof["overview"]]).T.reset_index().rename(columns={"index": "metric", 0: "value"}) \
            .to_excel(xw, sheet_name="Quality summary", index=False)
    return buf.getvalue()


def _parquet(df) -> bytes | None:
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    except Exception:
        return None


def render():
    ui.page_header("Export & report", "Take your cleaned data, the recipe that produced it, and a shareable report.")
    if not state.require_data():
        return
    s = st.session_state
    raw, df, steps = s.raw_df, state.current_df(), s.steps
    prof_after, prof_before = state.current_profile(), state.cached_profile(raw, s.token, state.steps_key([]))
    c = st.columns(3)
    with c[0]: ui.kpi("Quality before", f"{prof_before['overview']['quality_score']}/100")
    with c[1]: ui.kpi("Quality after", f"{prof_after['overview']['quality_score']}/100")
    with c[2]: ui.kpi("Steps applied", len(steps))
    st.write("")
    st.subheader("Download data")
    a, b, d, e = st.columns(4)
    base = (s.filename or "data").rsplit(".", 1)[0]
    a.download_button("⬇ CSV", df.to_csv(index=False).encode("utf-8"), f"{base}_cleaned.csv", "text/csv", width="stretch")
    b.download_button("⬇ Excel (3 sheets)", _excel(df, steps, prof_after), f"{base}_cleaned.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    pq = _parquet(df)
    d.download_button("⬇ Parquet", pq or b"", f"{base}_cleaned.parquet", disabled=pq is None, width="stretch")
    e.download_button("⬇ Pipeline JSON", json.dumps(steps, indent=2, default=str), "pipeline.json", "application/json",
                      width="stretch")
    st.caption("The Excel file includes the cleaning log and a quality summary on separate sheets.")

    st.subheader("Shareable report")
    st.caption("A single HTML file with the quality scores, cleaning log, insights, cautions and charts. "
               "Open it in a browser or print it to PDF. Charts load their library from the internet.")
    inc = st.checkbox("Include the latest predictive model summary", value=s.automl is not None, disabled=s.automl is None)
    if st.button("Build report", type="primary"):
        with st.spinner("Building report..."):
            extra = []
            html_doc = report.build_html(df, raw, steps, prof_before, prof_after, s.filename or "dataset", extra,
                                         s.automl.summary if inc and s.automl else None)
        st.session_state["report_html"] = html_doc
    if st.session_state.get("report_html"):
        st.download_button("⬇ Download report (HTML)", st.session_state["report_html"], f"{base}_report.html",
                           "text/html", type="primary")

    st.subheader("Save to history")
    name = st.text_input("Pipeline name", value="My cleaning pipeline")
    note = st.text_input("Tag or note (optional)", placeholder="e.g. Q3 sales, monthly refresh")
    if st.button("Save dataset and pipeline", disabled=not steps):
        try:
            state.db_ready()
            title = (name.strip() or "Untitled") + (f" [{note.strip()}]" if note.strip() else "")
            with SessionLocal() as db:
                if s.dataset_id is None:
                    s.dataset_id = save_dataset(db, s.filename, raw)
                save_pipeline(db, s.dataset_id, title, steps)
            st.toast("Saved to History")
        except Exception as e:
            st.error(f"Could not save to the database: {e}")
