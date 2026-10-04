"""Session state, cached computations and callbacks shared by every page."""
from __future__ import annotations

import json
import uuid

import pandas as pd
import streamlit as st

from core import eda
from core.clean import apply_step, replay
from core.profile import describe_step, profile, suggest_fixes
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


AUTO_SKIP_OPS = {"cap_outliers"}  # changes real values, so it stays a human decision


def auto_clean(max_passes: int = 6) -> dict:
    """Apply every safe suggested fix, repeating until nothing new shows up.

    Structural fixes (types, spellings, placeholders, dates) go first and gaps are filled afterwards. Outliers are
    left alone, and most-common-value filling is skipped for identity-like columns (names, emails, phone numbers),
    because that would invent data. Every step is recorded, so Undo and Reset still work."""
    s = st.session_state
    n0 = len(s.steps)
    before = current_profile()["overview"]["quality_score"]
    failed, left_alone = [], []
    for _ in range(max_passes):
        prof = current_profile()
        rows = max(prof["overview"]["rows"], 1)
        batch = []
        for sg in suggest_fixes(prof):
            if sg in failed or sg["op"] in AUTO_SKIP_OPS:
                continue
            if sg["op"] == "impute" and sg.get("method") == "mode" and prof["columns"][sg["col"]]["unique"] / rows > 0.5:
                if sg["col"] not in left_alone:
                    left_alone.append(sg["col"])
                continue
            batch.append(sg)
        batch = [b for b in batch if b["op"] != "impute"] or batch  # structure first, gap-filling last
        if not batch:
            break
        progressed = False
        for step in batch:
            n = len(s.steps)
            cb_apply(step)
            if len(s.steps) > n:
                progressed = True
            else:
                failed.append(step)
        if not progressed:
            break
    s.error = None
    after = current_profile()["overview"]
    left = [c for c, i in current_profile()["columns"].items() if i["outlier_pct"] > 1]
    return {"steps": len(s.steps) - n0, "before": before, "after": after["quality_score"],
            "left_alone": left_alone, "outlier_cols": left}


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
