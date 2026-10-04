"""Session state, cached computations and callbacks shared by every page."""
from __future__ import annotations

import json
import uuid

import pandas as pd
import streamlit as st

from core import eda
from core.clean import apply_step, replay
from core.profile import describe_step, profile
from db.models import init_db


def init_state():
    for k, v in {"raw_df": None, "filename": None, "steps": [], "redo": [], "last_upload_key": None,
                 "token": None, "dataset_id": None, "error": None, "pins": [], "automl": None,
                 "ask_history": []}.items():
        st.session_state.setdefault(k, v)


def set_dataset(df: pd.DataFrame, name: str, dataset_id: int | None = None):
    s = st.session_state
    s.raw_df, s.filename, s.steps, s.redo = df, name, [], []
    s.token, s.dataset_id, s.pins, s.automl, s.ask_history = uuid.uuid4().hex, dataset_id, [], None, []


def steps_key(steps=None) -> str:
    return json.dumps(st.session_state.steps if steps is None else steps, sort_keys=True, default=str)


# Heavy work is cached on (dataset token, steps) so clicking around does not re-run the pipeline.
@st.cache_data(show_spinner=False, max_entries=16)
def cached_replay(_raw, token: str, skey: str):
    return replay(_raw, json.loads(skey))


@st.cache_data(show_spinner=False, max_entries=32)
def cached_profile(_df, token: str, skey: str):
    return profile(_df)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_findings(_df, token: str, skey: str):
    return eda.generate_insights(_df), eda.guardrails(_df)


def current_df():
    raw = st.session_state.raw_df
    return None if raw is None else cached_replay(raw, st.session_state.token, steps_key())


def current_profile():
    return cached_profile(current_df(), st.session_state.token, steps_key())


@st.cache_resource
def db_ready() -> bool:
    init_db()
    return True


def require_data() -> bool:
    if st.session_state.raw_df is None:
        from app import ui
        ui.empty_state("No dataset yet", "Upload a file or try the sample on the Upload page to unlock this step.")
        return False
    return True


# ---- callbacks (run before the page re-renders, so the new state is shown straight away)
def cb_apply(step: dict):
    """Add a step only if it really works on the current data; otherwise show why not."""
    try:
        before = current_df()
        apply_step(before, step)
    except Exception as e:
        st.session_state.error = f"Could not apply '{describe_step(step)}': {e}"
        return
    st.session_state.error = None
    st.session_state.redo = []
    st.session_state.steps.append(step)


def cb_apply_all(steps: list):
    for s in steps:
        cb_apply(s)
        if st.session_state.error:
            break


def cb_undo():
    st.session_state.error = None
    if st.session_state.steps:
        st.session_state.redo.append(st.session_state.steps.pop())


def cb_redo():
    if st.session_state.redo:
        st.session_state.steps.append(st.session_state.redo.pop())


def cb_reset():
    st.session_state.error = None
    st.session_state.steps, st.session_state.redo = [], []


def pin(spec: dict):
    pins = st.session_state.pins
    if spec not in pins:
        pins.append(spec)
        st.toast("Pinned to the Dashboard")
    else:
        st.toast("Already on the Dashboard", icon="ℹ️")
