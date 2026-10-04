import pandas as pd
import plotly.express as px
import streamlit as st

from app import state, ui
from app.views.quality import KIND_LABEL, issues_for
from core import eda
from core.theme import style


def render():
    ui.page_header("Explore", "Patterns, distributions and relationships in your data.")
    if not state.require_data():
        return
    df, s = state.current_df(), st.session_state
    ui.chips([(f"Using cleaned data · {len(s.steps)} steps" if s.steps else "Using the original data · clean it first for better results",
               "ok" if s.steps else "warn")])
    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Column drill-down", "Correlations", "Chart builder", "Data table"])
    with t1:
        _overview(df)
    with t2:
        _drilldown(df)
    with t3:
        heat = eda.correlation_heatmap(df)
        if heat is None:
            st.info("Need at least two numeric columns.")
        else:
            st.plotly_chart(style(heat, 480), width="stretch")
            st.caption("Blue = rise together, red = one rises as the other falls. Correlation is not causation.")
    with t4:
        _builder(df)
    with t5:
        _table(df)


def _overview(df):
    s = st.session_state
    insights, warnings = state.cached_findings(df, s.token, state.steps_key())
    c = st.columns(4)
    with c[0]: ui.kpi("Rows", f"{len(df):,}")
    with c[1]: ui.kpi("Numeric columns", len(eda.numeric_cols(df)))
    with c[2]: ui.kpi("Category columns", len(eda.categorical_cols(df)))
    with c[3]: ui.kpi("Date columns", len(eda.datetime_cols(df)))
    st.write("")
    st.subheader("Key insights")
    for line in insights or ["No notable patterns found."]:
        st.markdown(f"- {line}")
    for w in warnings:
        st.warning(w)
    a, b = st.columns(2)
    with a:
        st.subheader("Numeric summary")
        st.dataframe(eda.summary_numeric(df), width="stretch")
    with b:
        st.subheader("Categorical summary")
        st.dataframe(eda.summary_categorical(df), hide_index=True, width="stretch")
    miss = eda.missing_matrix(df)
    if miss is not None:
        with st.expander("Where are values missing?"):
            st.plotly_chart(style(miss, 380), width="stretch")


def _drilldown(df):
    col = st.selectbox("Pick a column", list(df.columns), key="dd_col")
    s = df[col]
    from core.profile import profile_column
    info = profile_column(s, col)
    ui.chips([(KIND_LABEL.get(info["kind"], info["kind"]), "info")] + issues_for(info))
    c = st.columns(4)
    with c[0]: ui.kpi("Unique values", f"{s.nunique():,}")
    with c[1]: ui.kpi("Missing", f"{int(s.isna().sum()):,}", f"{s.isna().mean():.1%}")
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        with c[2]: ui.kpi("Mean", f"{s.mean():,.2f}", f"median {s.median():,.2f}")
        with c[3]: ui.kpi("Range", f"{s.min():,.2f} – {s.max():,.2f}")
    else:
        top = s.value_counts()
        if len(top):
            with c[2]: ui.kpi("Most common", str(top.index[0])[:18], f"{top.iloc[0]:,} rows")
    fig = eda.chart_for(df, col)
    if fig is not None:
        st.plotly_chart(style(fig, 360), width="stretch")
        st.button("Pin to dashboard", key="dd_pin", on_click=state.pin, args=({"kind": "chart", "x": col, "y": None},))
    if not pd.api.types.is_numeric_dtype(s):
        st.dataframe(s.value_counts(dropna=False).head(30).rename("rows").reset_index(), hide_index=True,
                     width="stretch")


def _builder(df):
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", cols, key="cb_x")
    y = c2.selectbox("Y axis (optional)", ["(none)"] + cols, key="cb_y")
    yy = None if y == "(none)" else y
    fig = eda.chart_for(df, x, yy)
    if fig is None:
        st.info("No suitable chart for that combination. Try a different pair.")
        return
    st.plotly_chart(style(fig, 440), width="stretch")
    st.button("Pin to dashboard", key="cb_pin", on_click=state.pin, args=({"kind": "chart", "x": x, "y": yy},))


def _table(df):
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("Search", placeholder="Filter rows by text in any column", key="tb_q")
    pick = c2.multiselect("Columns", list(df.columns), default=list(df.columns)[:12], key="tb_cols")
    view = df[pick] if pick else df
    if q:
        view = view[view.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)]
    st.caption(f"{len(view):,} rows")
    st.dataframe(view.head(1000), width="stretch", height=440)
