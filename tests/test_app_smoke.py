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
