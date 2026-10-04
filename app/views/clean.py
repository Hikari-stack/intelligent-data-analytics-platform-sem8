import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app import state, ui
from app.views.quality import render_profile
from core import eda, outliers
from core.clean import FILTER_OPERATORS, replay
from core.profile import describe_step, suggest_fixes
from core.theme import style

TOOLS = ["Fill missing values", "Filter rows", "Find & replace text", "Convert column type", "Rename column",
         "Drop column", "Remove duplicates"]


def render():
    ui.page_header("Clean", "Apply fixes one at a time. Nothing changes until you click, and everything can be undone.")
    if not state.require_data():
        return
    s = st.session_state
    raw, df, steps = s.raw_df, state.current_df(), s.steps
    if s.error:
        st.error(s.error)
        s.error = None

    left, right = st.columns([3, 2], gap="large")
    with left:
        t1, t2, t3 = st.tabs(["Suggested fixes", "Manual tools", "Outliers"])
        with t1:
            _suggestions(df, steps)
        with t2:
            _manual(df)
        with t3:
            _outliers(df)
    with right:
        _pipeline_panel(raw, steps)

    st.divider()
    st.subheader("Before vs after")
    ba, bb = st.columns(2, gap="large")
    with ba:
        st.caption("BEFORE")
        render_profile(raw, "Original", state.steps_key([]), compact=True)
    with bb:
        st.caption("AFTER")
        render_profile(df, "Cleaned", state.steps_key(), compact=True)

    num_cols = [c for c in raw.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        col = st.selectbox("Compare a numeric column", num_cols)
        raw_num = pd.to_numeric(raw[col], errors="coerce")
        c1, c2 = st.columns(2)
        c1.plotly_chart(style(px.histogram(raw_num.dropna(), title=f"{col} (before)"), 300), width="stretch")
        c2.plotly_chart(style(px.histogram(df[col].dropna(), title=f"{col} (after)"), 300), width="stretch")


def _suggestions(df, steps):
    suggestions = suggest_fixes(state.current_profile())
    if not suggestions:
        st.success("No more issues found. Your data looks clean.")
        return
    st.caption("After applying, new suggestions may appear (e.g. filling missing values once a column is converted).")
    st.button(f"Apply all {len(suggestions)} suggestions", on_click=state.cb_apply_all, args=(suggestions,),
              type="primary", width="stretch")
    for idx, sg in enumerate(suggestions):
        with st.container(border=True):
            a, b = st.columns([4, 1])
            a.markdown(f"**{describe_step(sg)}**")
            a.caption(sg["reason"])
            b.button("Apply", key=f"apply_{len(steps)}_{idx}", on_click=state.cb_apply, args=(sg,),
                     width="stretch")


def _manual(df):
    tool = st.selectbox("Tool", TOOLS, key="manual_tool")
    cols = list(df.columns)
    step = None
    if tool == "Remove duplicates":
        st.caption(f"{int(df.duplicated().sum())} fully duplicated rows right now.")
        step = {"op": "drop_duplicates"}
    elif tool == "Drop column":
        step = {"op": "drop_column", "col": st.selectbox("Column to drop", cols, key="m_drop")}
    elif tool == "Rename column":
        c = st.selectbox("Column", cols, key="m_ren")
        step = {"op": "rename_column", "col": c, "new_name": st.text_input("New name", key="m_ren_to")}
    elif tool == "Fill missing values":
        withna = [c for c in cols if df[c].isna().any()] or cols
        c = st.selectbox("Column", withna, key="m_fill")
        is_num = pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        methods = (["median", "mean", "knn", "mode", "constant"] if is_num else ["mode", "constant"])
        m = st.selectbox("Method", methods, key="m_fill_m",
                         help="knn = copy values from the most similar rows, judged on the other numeric columns.")
        step = {"op": "impute", "col": c, "method": m}
        if m == "constant":
            v = st.text_input("Value", key="m_fill_v")
            step["value"] = (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() and is_num else v)
        if m == "knn":
            step["k"] = st.slider("Neighbours (k)", 1, 15, 5, key="m_fill_k")
    elif tool == "Filter rows":
        c = st.selectbox("Column", cols, key="m_flt")
        a, b = st.columns(2)
        op = a.selectbox("Keep rows where it", FILTER_OPERATORS, key="m_flt_op")
        val = b.text_input("Value", key="m_flt_v", disabled=op in ("is missing", "is not missing"))
        step = {"op": "filter_rows", "col": c, "operator": op, "value": val}
    elif tool == "Find & replace text":
        texts = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])] or cols
        c = st.selectbox("Column", texts, key="m_fr")
        a, b = st.columns(2)
        step = {"op": "find_replace", "col": c, "find": a.text_input("Find", key="m_fr_f"),
                "replace": b.text_input("Replace with", key="m_fr_r")}
    elif tool == "Convert column type":
        c = st.selectbox("Column", cols, key="m_cv")
        kind = st.selectbox("Convert to", ["numbers", "dates", "True/False", "standardized categories"], key="m_cv_k")
        step = {"op": {"numbers": "convert_numeric", "dates": "convert_datetime", "True/False": "convert_boolean",
                       "standardized categories": "standardize_categories"}[kind], "col": c}
    st.button("Add to pipeline", on_click=state.cb_apply, args=(step,), type="primary", key="m_add")


def _outliers(df):
    summ = outliers.column_summary(df)
    if summ.empty:
        st.info("No numeric columns with enough variety to check for outliers.")
        return
    st.dataframe(summ, hide_index=True, width="stretch")
    worst = int(summ["iqr_outliers"].values.argmax())
    c = st.selectbox("Inspect a column", list(summ["column"]), index=worst, key="o_col")
    row = summ[summ.column == c].iloc[0]
    st.plotly_chart(style(px.box(df, x=c, points="outliers", title=f"{c}: outliers marked"), 220), width="stretch")
    a, b, m = st.columns([1, 1, 1])
    method = m.selectbox("Rule", ["iqr", "zscore"], key="o_rule", label_visibility="collapsed")
    none = int(row.iqr_outliers) == 0 and method == "iqr"
    a.button(f"Cap {int(row.iqr_outliers)} at limits", on_click=state.cb_apply, disabled=none,
             args=({"op": "cap_outliers", "col": c},), width="stretch", key="o_cap",
             help="Pull extreme values back to the IQR limits; keeps every row.")
    b.button("Remove those rows", on_click=state.cb_apply, disabled=none,
             args=({"op": "drop_outlier_rows", "col": c, "method": method},), width="stretch", key="o_drop",
             help="Deletes the rows entirely. Use when the values are errors.")
    with st.expander("Unusual rows across all columns together (Isolation Forest)"):
        st.caption("A row can look normal in each column but be strange as a combination. This view is for review only.")
        rate = st.slider("Share of rows to flag", 0.5, 10.0, 2.0, 0.5, format="%.1f%%", key="o_rate")
        flagged = outliers.isolation_outliers(df, rate / 100)
        if flagged.empty:
            st.write("Not enough numeric data (needs 20+ rows).")
        else:
            st.dataframe(flagged.head(100), width="stretch", height=260)


def _pipeline_panel(raw, steps):
    s = st.session_state
    st.subheader("Pipeline")
    ui.timeline([describe_step(x) for x in steps])
    a, b, c = st.columns(3)
    a.button("↩ Undo", on_click=state.cb_undo, disabled=not steps, width="stretch")
    b.button("↪ Redo", on_click=state.cb_redo, disabled=not s.redo, width="stretch")
    c.button("Reset", on_click=state.cb_reset, disabled=not (steps or s.redo), width="stretch")
    with st.expander("Replay a saved pipeline on this dataset"):
        log_file = st.file_uploader("Pipeline JSON", type=["json"], key="logfile")
        if log_file is not None and st.button("Replay uploaded pipeline"):
            try:
                loaded = json.load(log_file)
                replay(raw, loaded)  # dry run: raises if a column is missing
                s.steps, s.redo = loaded, []
                st.rerun()
            except Exception as e:
                st.error(f"Could not replay this pipeline on this dataset: {e}")
    st.download_button("⬇ Download pipeline (JSON)", json.dumps(steps, indent=2, default=str),
                       file_name="pipeline.json", mime="application/json", width="stretch")
