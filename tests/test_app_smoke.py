"""Drive the whole Streamlit app headlessly: every page must render without an exception."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
PAGES = ["Upload", "Quality report", "Clean", "Explore", "Insights",
         "Dashboard", "Export & report", "History"]


def _go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    at.toggle(key="auto_clean").set_value(False).run()   # these tests drive the cleaning steps by hand
    at.button[0].click().run()          # "Use the sample messy dataset"
    assert not at.exception
    return at


def test_empty_state_pages_render():
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not at.exception
    for p in PAGES:
        _go(at, p)


def test_every_page_with_data(app):
    for p in PAGES:
        _go(app, p)


def test_apply_all_suggestions_then_pages(app):
    _go(app, "Clean")
    next(b for b in app.button if b.label.startswith("Apply all")).click().run()
    assert not app.exception
    assert len(app.session_state.steps) > 3
    for p in PAGES:
        _go(app, p)
    # undo / redo round trip
    _go(app, "Clean")
    n = len(app.session_state.steps)
    next(b for b in app.button if "Undo" in b.label).click().run()
    assert len(app.session_state.steps) == n - 1
    next(b for b in app.button if "Redo" in b.label).click().run()
    assert len(app.session_state.steps) == n


def test_ask_automl_manual_tool_and_report(app):
    _go(app, "Clean")
    next(b for b in app.button if b.label.startswith("Apply all")).click().run()
    # manual tool: filter rows
    app.selectbox(key="manual_tool").set_value("Filter rows").run()
    app.selectbox(key="m_flt").set_value("Quantity").run()
    app.selectbox(key="m_flt_op").set_value(">=").run()
    app.text_input(key="m_flt_v").set_value("3").run()
    n = len(app.session_state.steps)
    app.button(key="m_add").click().run()
    assert not app.exception and len(app.session_state.steps) == n + 1
    # ask your data
    _go(app, "Insights")
    app.text_input(key="ask_q").set_value("average sales by region")
    next(b for b in app.button if b.label == "Ask").click().run()
    assert not app.exception
    app.button(key="am_go").click().run()
    assert not app.exception and app.session_state.automl is not None
    # pin + dashboard
    app.button(key="ask_pin").click().run()
    assert app.session_state.pins
    _go(app, "Dashboard")
    # report
    _go(app, "Export & report")
    next(b for b in app.button if "Build report" in b.label).click().run()
    assert not app.exception and "<html" in app.session_state["report_html"]


def test_bad_manual_step_shows_error_not_crash(app):
    _go(app, "Clean")
    app.selectbox(key="manual_tool").set_value("Filter rows").run()
    app.selectbox(key="m_flt").set_value("Quantity").run()
    app.text_input(key="m_flt_v").set_value("not a number").run()
    app.button(key="m_add").click().run()
    assert not app.exception and app.session_state.steps == []
    assert app.error


def test_auto_clean_on_load_applies_safe_steps_and_pages_render():
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    at.button[0].click().run()          # auto-clean is on by default
    assert not at.exception
    steps = at.session_state.steps
    assert steps, "auto-clean should have applied steps"
    assert not any(st["op"] == "cap_outliers" for st in steps), "outliers must be left for a human"
    assert at.session_state.clean_notice and "Cleaned automatically" in at.session_state.clean_notice
    for p in PAGES:
        _go(at, p)


def test_auto_clean_does_not_invent_identity_values():
    import io
    import pandas as pd
    import streamlit as st
    from app import state

    n = 60
    raw = pd.DataFrame({
        "email": [f"user{i}@x.com" if i % 6 else None for i in range(n)],      # unique per row, some missing
        "city": ["Pune", "Delhi", " pune ", "DELHI", None, "Goa"] * 10,
        "amount": [str(i * 10) if i % 7 else "N/A" for i in range(n)],
    })
    at = AppTest.from_function(lambda: None).run()
    st.session_state.clear()
    state.init_state()
    state.set_dataset(raw, "t.csv")
    result = state.auto_clean()
    cols = {s.get("col") for s in st.session_state.steps if s["op"] == "impute"}
    assert "email" not in cols and "email" in result["left_alone"]
    assert "city" in cols or any(s["op"] == "standardize_categories" for s in st.session_state.steps)
