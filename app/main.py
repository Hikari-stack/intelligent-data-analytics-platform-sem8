"""Intelligent Data Analytics Platform - Streamlit app.

Run from the project root:   streamlit run app/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `import core...` and `import app...` work when launched by Streamlit

import streamlit as st

st.set_page_config(page_title="Intelligent Analytics", page_icon="📊", layout="wide")

from app import state, ui
from app.views import clean, dashboard, explore, export, history, insights, quality, upload
import core.theme  # noqa: F401  (registers the shared Plotly template)

PAGES = {  # label -> (module, stepper index or -1)
    "📥  Upload": (upload, 0),
    "🩺  Quality report": (quality, 1),
    "🧹  Clean": (clean, 2),
    "🔎  Explore": (explore, 3),
    "🧠  Insights": (insights, 4),
    "📌  Dashboard": (dashboard, -1),
    "📦  Export & report": (export, 5),
    "🕘  History": (history, -1),
}

state.init_state()
ui.inject_css()
ui.sidebar_brand()
page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")

s = st.session_state
if s.raw_df is not None:
    df = state.current_df()
    ui.sidebar_dataset(s.filename, len(df), df.shape[1], len(s.steps), state.current_profile()["overview"]["quality_score"])
else:
    ui.sidebar_dataset(None, None, None, 0, None)

module, idx = PAGES[page]
if idx >= 0:
    ui.stepper(idx, s.raw_df is not None, bool(s.steps))
module.render()
