import pandas as pd
import streamlit as st

from app import state, ui
from core import eda, nlq
from core.theme import style


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Dashboard filters", expanded=False):
        dts, cats = eda.datetime_cols(df), [c for c in eda.categorical_cols(df) if 2 <= df[c].nunique() <= 30][:3]
        if dts:
            d = dts[0]
            lo, hi = df[d].min(), df[d].max()
            if pd.notna(lo) and pd.notna(hi) and lo < hi:
                rng = st.date_input(f"{d} range", (lo.date(), hi.date()), min_value=lo.date(), max_value=hi.date(), key="db_dr")
                if isinstance(rng, tuple) and len(rng) == 2:
                    df = df[(df[d] >= pd.Timestamp(rng[0])) & (df[d] <= pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]
        for c in cats:
            picked = st.multiselect(c, sorted(df[c].dropna().unique().tolist(), key=str), key=f"db_f_{c}")
            if picked:
                df = df[df[c].isin(picked)]
    return df


def render():
    ui.page_header("Dashboard", "Pin charts and answers from Explore and Insights to build your own view.")
    if not state.require_data():
        return
    s = st.session_state
    df = state.current_df()
    if not s.pins:
        ui.empty_state("Nothing pinned yet",
                       "Use the Pin button under any chart in Explore, or under an answer in Ask your data.")
    view = _filters(df)
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    nums = eda.numeric_cols(view)[:4]
    if nums:
        c = st.columns(len(nums))
        for col, n in zip(c, nums):
            with col:
                ui.kpi(n, f"{view[n].sum():,.0f}" if view[n].abs().max() > 100 else f"{view[n].mean():,.2f}",
                       "total" if view[n].abs().max() > 100 else "average")
        st.write("")
    left = True
    cols = st.columns(2, gap="large")
    for i, p in enumerate(list(s.pins)):
        with cols[i % 2]:
            fig, title = _render_pin(view, p)
            with st.container(border=True):
                if fig is None:
                    st.warning(f"Could not draw {title}. A column may have been removed by cleaning.")
                else:
                    st.plotly_chart(style(fig, 320), width="stretch", key=f"pin_{i}")
                if st.button("Remove", key=f"unpin_{i}"):
                    s.pins.pop(i)
                    st.rerun()


def _render_pin(df, p):
    try:
        if p["kind"] == "ask":
            a = nlq.ask(df, p["q"])
            return a.figure, p["q"]
        return eda.chart_for(df, p["x"], p.get("y")), f"{p['x']} / {p.get('y')}"
    except Exception:
        return None, str(p)
