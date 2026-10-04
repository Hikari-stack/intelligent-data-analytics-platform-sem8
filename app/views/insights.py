import pandas as pd
import plotly.express as px
import streamlit as st

from app import state, ui
from core import automl, eda, nlq, timeseries
from core.theme import ACCENT, style


def render():
    ui.page_header("Insights", "Ask questions, study trends over time, and see what predicts what.")
    if not state.require_data():
        return
    df = state.current_df()
    t1, t2, t3 = st.tabs(["Ask your data", "Time series", "Predict (AutoML)"])
    with t1:
        _ask(df)
    with t2:
        _time(df)
    with t3:
        _automl(df)


def _ask(df):
    s = st.session_state
    st.caption("Plain English in, answer + chart + code out. Runs entirely on your machine, no external service.")
    ex = nlq._suggest(df)
    st.markdown("Try: " + " ".join(f"`{e}`" for e in ex[:4]))
    with st.form("ask_form", clear_on_submit=False):
        q = st.text_input("Your question", placeholder="e.g. which region has the highest sales?", key="ask_q")
        go = st.form_submit_button("Ask", type="primary")
    if go and q.strip():
        s.ask_history = [q.strip()] + [h for h in s.ask_history if h != q.strip()][:7]
    if not (q and q.strip()):
        if s.ask_history:
            st.caption("Recent: " + " · ".join(s.ask_history[:5]))
        return
    a = nlq.ask(df, q)
    if not a.ok:
        st.warning(a.text)
        st.markdown("Examples that work: " + ", ".join(f"`{e}`" for e in a.suggestions))
        return
    ui.answer_box(a.text)
    if a.figure is not None:
        st.plotly_chart(style(a.figure, 380), width="stretch")
    if a.table is not None:
        with st.expander("Show the numbers"):
            st.dataframe(a.table, hide_index=True, width="stretch")
    with st.expander("Show the code behind this answer"):
        st.code(a.code, language="python")
    if a.figure is not None:
        st.button("Pin answer to dashboard", on_click=state.pin, args=({"kind": "ask", "q": q.strip()},), key="ask_pin")


def _time(df):
    dts = eda.datetime_cols(df)
    if not dts:
        st.info("This tab needs a date column. Convert one on the Clean page (look for 'dates stored as text').")
        return
    nums = eda.numeric_cols(df)
    c = st.columns(4)
    d = c[0].selectbox("Date column", dts, key="ts_d")
    v = c[1].selectbox("Measure", ["(count rows)"] + nums, key="ts_v")
    freq = c[2].selectbox("Period", list(timeseries.FREQS), index=1, key="ts_f")
    agg = c[3].selectbox("Combine by", timeseries.AGGS, disabled=v == "(count rows)", key="ts_a")
    series = timeseries.aggregate(df, d, None if v == "(count rows)" else v, freq, agg)
    if len(series) < 2:
        st.info("Not enough periods. Try a finer period such as Week or Day.")
        return
    win = st.slider("Moving-average window (periods)", 2, 12, 3, key="ts_w")
    res = timeseries.analyse(series, win)
    for line in res["summary"]:
        st.markdown(f"- {line}")
    st.plotly_chart(style(timeseries.figure(res, f"{agg.title() if v != '(count rows)' else 'Count'} per {freq.lower()}"), None),
                    width="stretch")


def _automl(df):
    targets = automl.candidate_targets(df)
    if not targets:
        st.info("No suitable target column found (need a column with 2+ values and little missing data).")
        return
    st.caption("Pick a column to predict. The app trains simple models, compares them with a naive guess, "
               "and warns you when the result should not be trusted.")
    c = st.columns([2, 1])
    t = c[0].selectbox("What do you want to predict?", targets, key="am_t")
    task = automl.detect_task(df[t])
    c[1].markdown(f"<br>{ui.chip('classification' if task == 'classification' else 'regression (number)', 'info')}",
                  unsafe_allow_html=True)
    if st.button("Train models", type="primary", key="am_go"):
        with st.spinner("Training and validating models..."):
            try:
                st.session_state.automl = automl.run(df, t)
            except ValueError as e:
                st.session_state.automl = None
                st.error(str(e))
    res = st.session_state.automl
    if res is None:
        return
    if res.target != t:
        st.caption(f"Showing results for the previous run (target: {res.target}).")
    for line in res.summary:
        st.markdown(f"- {line}")
    for w in res.warnings:
        st.warning(w)
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.subheader("Model comparison")
        st.dataframe(res.leaderboard, hide_index=True, width="stretch")
        st.caption(f"Trained on {res.n_train:,} rows, tested on {res.n_test:,} unseen rows. 'cv' = cross-validation score.")
    with b:
        st.subheader("What mattered most")
        imp = res.importances.head(10).iloc[::-1]
        fig = px.bar(imp, x="importance", y="feature", orientation="h")
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(style(fig, 320), width="stretch")
    with st.expander("Predictions on the unseen test rows"):
        st.dataframe(res.predictions.head(200), width="stretch")
