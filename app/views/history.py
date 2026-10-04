import pandas as pd
import streamlit as st

from app import state, ui
from core.clean import replay
from db.repo import list_datasets, list_pipelines, load_dataset, load_pipeline_steps
from db.session import SessionLocal


def render():
    ui.page_header("History", "Datasets and cleaning pipelines you saved earlier.")
    try:
        state.db_ready()
        with SessionLocal() as db:
            datasets = list_datasets(db)
            if not datasets:
                ui.empty_state("🕘", "Nothing saved yet", "Clean a dataset, then use Save to history on the Export page.")
                return
            rows = pd.DataFrame([{"id": d.id, "name": d.name, "rows": d.n_rows, "columns": d.n_cols,
                                  "saved": d.created_at.strftime("%Y-%m-%d %H:%M")} for d in datasets])
            st.dataframe(rows, hide_index=True, width="stretch")

            labels = {d.id: f"#{d.id}  {d.name}  ({d.n_rows:,} rows)" for d in datasets}
            ds_id = st.selectbox("Dataset", list(labels), format_func=labels.get)
            pipelines = list_pipelines(db, ds_id)
            pl_labels = {p.id: f"{p.name}  ({len(p.steps)} steps, {p.created_at:%Y-%m-%d %H:%M})" for p in pipelines}
            pl_id = None
            if pipelines:
                pl_id = st.selectbox("Pipeline (optional)", [None] + list(pl_labels),
                                     format_func=lambda i: "(none, just load the data)" if i is None else pl_labels[i])
            else:
                st.caption("No pipelines saved for this dataset.")

            if st.button("Load", type="primary"):
                df = load_dataset(db, ds_id)
                steps = load_pipeline_steps(db, pl_id) if pl_id is not None else []
                replay(df, steps)  # dry run: raises if the pipeline does not fit
                state.set_dataset(df, labels[ds_id].split("  ")[1], dataset_id=ds_id)
                st.session_state.steps = list(steps)
                st.session_state.last_upload_key = None
                st.success("Loaded. Continue on the Clean or Explore page.")
    except Exception as e:
        st.error(f"Could not read the history: {e}")
