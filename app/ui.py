"""Visual layer: CSS, header, stepper, cards, gauge. No business logic lives here."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from core.theme import ACCENT, BAD, GOOD, INK, LINE, MUTED, WARN

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
:root {
  --ink:#1F2933; --muted:#5F6B76; --line:#DAD6CE; --paper:#FBFAF7; --panel:#F3F1EB;
  --accent:#1E5A78; --accent-soft:#E4EEF3;
  --good:#3F7D58; --warn:#B7791F; --bad:#B3412F;
}
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1200px; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); }
h2, h3 { font-size: 1.3rem; }

/* sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] hr { border-color: var(--line); }
section[data-testid="stSidebar"] .stRadio label { padding: 5px 10px; border-radius: 4px; width: 100%; font-size: 14.5px; }
section[data-testid="stSidebar"] .stRadio label:hover { background: #E8E5DC; }
.brand { margin: 2px 0 18px; padding-bottom: 14px; border-bottom: 1px solid var(--line); }
.brand b { font-family: 'Source Serif 4', Georgia, serif; font-size: 18px; font-weight: 600; display:block; line-height:1.2; }
.brand span { font-size: 12.5px; color: var(--muted); }
.side-card { border:1px solid var(--line); background: var(--paper); border-radius:4px; padding:10px 12px; margin-top:12px; font-size:13px; line-height:1.5; }
.side-card small { color: var(--muted); }

/* page header */
.page-head { margin-bottom: 1.2rem; }
.page-head h1 { font-size: 2rem; margin: 0; }
.page-head p { color: var(--muted); margin: 6px 0 0; font-size: 1rem; max-width: 720px; }

/* stepper */
.stepper { display:flex; margin: 0 0 1.6rem; flex-wrap: wrap; border-bottom: 1px solid var(--line); }
.step { padding: 8px 18px 10px 0; margin-right: 18px; font-size: 13.5px; color: var(--muted);
  display:flex; align-items:center; gap:7px; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.step i { font-style: normal; font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted); }
.step.done { color: var(--ink); } .step.done i { color: var(--good); }
.step.now { color: var(--ink); font-weight: 600; border-bottom-color: var(--accent); }
.step.now i { color: var(--accent); }
.step.lock { opacity: .5; }

/* cards and panels */
.card { border-top: 2px solid var(--ink); padding: 12px 2px 4px; height:100%; }
.card h4 { margin:0 0 4px; font-size:15.5px; font-weight:600; font-family:'IBM Plex Sans',sans-serif; }
.card p { margin:0; color: var(--muted); font-size:14px; line-height:1.5; }
.card.empty { border:1px dashed var(--line); border-radius:4px; text-align:center; padding:40px 20px; }
.card.empty h4 { font-family:'Source Serif 4', Georgia, serif; font-size:18px; }
.intro { padding: 4px 0 22px; margin-bottom: 1.4rem; border-bottom: 1px solid var(--line); }
.intro h1 { font-size: 2.3rem; margin: 0 0 10px; line-height: 1.15; }
.intro p { color: var(--muted); font-size: 1.05rem; max-width: 660px; margin: 0; line-height: 1.55; }
.chip { display:inline-block; padding:1px 8px; border-radius:3px; font-size:12px; font-weight:500; margin:2px 4px 2px 0; border:1px solid transparent; }
.chip.ok { background:#E6F0EA; color:#2F6244; } .chip.warn { background:#F8EEDB; color:#8A5A12; }
.chip.bad { background:#F5E3DF; color:#8E2F20; } .chip.info { background: var(--accent-soft); color: var(--accent); }
.chip.mute { background:#EDEAE2; color:#4A545E; }
.kpi { border:1px solid var(--line); border-radius:4px; padding:12px 14px; background:#fff; min-height:98px; margin-bottom:14px; }
.score { border:1px solid var(--line); border-radius:4px; background:#fff; padding:16px 20px; min-height:98px; margin-bottom:14px; }
.score span { font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.score b { display:block; font-family:'Source Serif 4',Georgia,serif; font-size:2.6rem; line-height:1.1; margin:2px 0 10px; font-variant-numeric:tabular-nums; }
.score b small { font-size:1rem; color:var(--muted); font-family:'IBM Plex Sans',sans-serif; font-weight:400; }
.score .bar { height:6px; background:#ECE9E2; border-radius:3px; overflow:hidden; }
.score .bar div { height:100%; }
.score em { display:block; font-style:normal; font-size:13px; color:var(--muted); margin-top:8px; }
.kpi span { font-size:11.5px; color: var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.kpi b { display:block; font-size:1.5rem; margin-top:2px; font-weight:600; font-variant-numeric: tabular-nums; }
.kpi small { color: var(--muted); }
.timeline { border-left:1px solid var(--line); margin-left:6px; padding-left:16px; }
.timeline .t { position:relative; margin:0 0 10px; font-size:14px; }
.timeline .t::before { content:''; position:absolute; left:-21px; top:7px; width:7px; height:7px; border-radius:50%; background: var(--accent); }
.answer { border-left:3px solid var(--accent); background: var(--panel); padding:12px 16px; border-radius:0 4px 4px 0; font-size:1.02rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:4px; padding:10px 14px; }
.stButton > button, .stDownloadButton > button { border-radius:4px; font-weight:500; }
div[data-testid="stAlert"] { border-radius:4px; }
div[data-testid="stTabs"] button { font-weight:500; }
</style>
"""

STEPS = ["Upload", "Quality", "Clean", "Explore", "Insights", "Export"]


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown("<div class='brand'><b>Intelligent Data Analytics</b>"
                        "<span>Check, clean and explain your data</span></div>", unsafe_allow_html=True)


def sidebar_dataset(name: str | None, rows: int | None, cols: int | None, steps: int, score: float | None):
    if name is None:
        st.sidebar.markdown("<div class='side-card'>No dataset loaded<br><small>Start on the Upload page</small></div>",
                            unsafe_allow_html=True)
        return
    st.sidebar.markdown(
        f"<div class='side-card'><b>{html.escape(name)}</b><br><small>{rows:,} rows · {cols} columns</small><br>"
        f"<small>{steps} cleaning step{'s' if steps != 1 else ''} · quality {score if score is not None else '–'}/100</small></div>",
        unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"<div class='page-head'><h1>{html.escape(title)}</h1>"
                f"{f'<p>{html.escape(subtitle)}</p>' if subtitle else ''}</div>", unsafe_allow_html=True)


def intro(title: str, text: str):
    st.markdown(f"<div class='intro'><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>",
                unsafe_allow_html=True)


def stepper(current: int, has_data: bool, cleaned: bool):
    out = []
    for i, label in enumerate(STEPS):
        done = (i == 0 and has_data) or (i == 2 and cleaned) or (has_data and i < current and i != 2)
        cls = "now" if i == current else "done" if done else ("lock" if not has_data and i > 0 else "")
        mark = "✓" if done and i != current else f"{i + 1}."
        out.append(f"<div class='step {cls}'><i>{mark}</i>{label}</div>")
    st.markdown(f"<div class='stepper'>{''.join(out)}</div>", unsafe_allow_html=True)


def card(title: str, text: str):
    st.markdown(f"<div class='card'><h4>{html.escape(title)}</h4><p>{html.escape(text)}</p></div>",
                unsafe_allow_html=True)


def kpi(label: str, value, sub: str = ""):
    st.markdown(f"<div class='kpi'><span>{html.escape(label)}</span><b>{html.escape(str(value))}</b>"
                f"<small>{html.escape(sub)}</small></div>", unsafe_allow_html=True)


def chip(text: str, kind: str = "mute") -> str:
    return f"<span class='chip {kind}'>{html.escape(text)}</span>"


def chips(items: list[tuple[str, str]]):
    st.markdown("".join(chip(t, k) for t, k in items), unsafe_allow_html=True)


def score_block(score: float, title: str = "Quality score"):
    """Plain score readout with a thin progress bar and a one-line verdict."""
    color = GOOD if score >= 80 else WARN if score >= 55 else BAD
    verdict = "Good shape" if score >= 80 else "Usable, but needs cleaning" if score >= 55 else "Needs significant cleaning"
    st.markdown(f"<div class='score'><span>{html.escape(title)}</span>"
                f"<b style='color:{color}'>{score:g}<small> / 100</small></b>"
                f"<div class='bar'><div style='width:{max(0, min(100, score))}%;background:{color}'></div></div>"
                f"<em>{verdict}</em></div>", unsafe_allow_html=True)


def gauge(score: float, title: str = "Quality score", height: int = 210):
    """Horizontal bullet gauge: easier to read than a dial and closer to a report figure."""
    color = GOOD if score >= 80 else WARN if score >= 55 else BAD
    fig = go.Figure(go.Indicator(
        mode="number+gauge", value=score, number={"suffix": " / 100", "font": {"size": 34, "color": INK}},
        title={"text": title, "font": {"size": 14, "color": MUTED}},
        gauge={"shape": "bullet", "axis": {"range": [0, 100], "tickvals": [0, 55, 80, 100],
                                           "tickfont": {"size": 11, "color": MUTED}},
               "bar": {"color": color, "thickness": 0.45}, "bgcolor": "#fff", "borderwidth": 0,
               "steps": [{"range": [0, 55], "color": "#F1E1DC"}, {"range": [55, 80], "color": "#F3EAD3"},
                         {"range": [80, 100], "color": "#E2EDE5"}]}))
    fig.update_layout(height=min(height, 170), margin=dict(l=20, r=30, t=50, b=30), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def timeline(items: list[str]):
    if not items:
        st.caption("No steps applied yet.")
        return
    st.markdown("<div class='timeline'>" + "".join(f"<div class='t'>{html.escape(t)}</div>" for t in items) + "</div>",
                unsafe_allow_html=True)


def answer_box(md_text: str):
    import re
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(md_text))
    st.markdown(f"<div class='answer'>{safe}</div>", unsafe_allow_html=True)


def empty_state(title: str, text: str):
    st.markdown(f"<div class='card empty'><h4>{html.escape(title)}</h4><p>{html.escape(text)}</p></div>",
                unsafe_allow_html=True)
