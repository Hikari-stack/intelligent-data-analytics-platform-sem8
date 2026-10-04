"""Visual layer: CSS, header, stepper, cards, gauge. No business logic lives here."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from core.theme import ACCENT, BAD, GOOD, LINE, MUTED, WARN

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 650; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0F172A; }
section[data-testid="stSidebar"] * { color: #E2E8F0; }
section[data-testid="stSidebar"] hr { border-color: #1E293B; }
section[data-testid="stSidebar"] .stRadio label { padding: 6px 10px; border-radius: 8px; width: 100%; }
section[data-testid="stSidebar"] .stRadio label:hover { background: #1E293B; }
.brand { display:flex; align-items:center; gap:10px; margin: 4px 0 14px; }
.brand .logo { width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,#6366F1,#06B6D4);
  display:flex; align-items:center; justify-content:center; font-weight:700; color:#fff !important; }
.brand b { font-size: 15px; line-height:1.1; } .brand span { font-size: 11px; color:#94A3B8 !important; }
.side-card { background:#1E293B; border-radius:12px; padding:12px 14px; margin-top:10px; font-size:13px; }
.side-card small { color:#94A3B8 !important; }

/* page header */
.page-head { margin-bottom: 1.1rem; }
.page-head h1 { font-size: 1.85rem; margin: 0; }
.page-head p { color: #64748B; margin: 4px 0 0; font-size: 0.97rem; }

/* stepper */
.stepper { display:flex; gap:6px; margin: 0 0 1.4rem; flex-wrap: wrap; }
.step { flex:1; min-width:120px; padding:9px 12px; border-radius:10px; border:1px solid #E2E8F0; background:#fff;
  font-size:13px; display:flex; align-items:center; gap:8px; color:#64748B; }
.step i { font-style:normal; width:22px; height:22px; border-radius:50%; background:#F1F5F9; display:flex;
  align-items:center; justify-content:center; font-size:11px; font-weight:600; flex:none; }
.step.done { color:#0F172A; } .step.done i { background:#D1FAE5; color:#047857; }
.step.now { border-color:#4F46E5; box-shadow:0 0 0 3px #EEF2FF; color:#0F172A; font-weight:600; }
.step.now i { background:#4F46E5; color:#fff; } .step.lock { opacity:.55; }

/* cards */
.card { border:1px solid #E2E8F0; border-radius:14px; padding:16px 18px; background:#fff; height:100%;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.card h4 { margin:0 0 4px; font-size:15px; } .card p { margin:0; color:#64748B; font-size:13.5px; }
.card .ic { font-size:22px; margin-bottom:8px; }
.hero { border-radius:18px; padding:34px 34px 30px; margin-bottom:1.2rem; color:#fff;
  background: radial-gradient(1200px 400px at 90% -20%, #22D3EE55, transparent), linear-gradient(135deg,#312E81,#4F46E5 55%,#6366F1); }
.hero h1 { color:#fff; font-size:2.1rem; margin:0 0 8px; } .hero p { color:#E0E7FF; font-size:1.02rem; max-width:680px; margin:0; }
.chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:500; margin:2px 4px 2px 0; }
.chip.ok { background:#D1FAE5; color:#047857; } .chip.warn { background:#FEF3C7; color:#B45309; }
.chip.bad { background:#FEE2E2; color:#B91C1C; } .chip.info { background:#E0E7FF; color:#4338CA; }
.chip.mute { background:#F1F5F9; color:#475569; }
.kpi { border:1px solid #E2E8F0; border-radius:14px; padding:14px 16px; background:#fff; }
.kpi span { font-size:12px; color:#64748B; text-transform:uppercase; letter-spacing:.05em; }
.kpi b { display:block; font-size:1.55rem; margin-top:2px; letter-spacing:-.02em; }
.kpi small { color:#64748B; }
.timeline { border-left:2px solid #E2E8F0; margin-left:8px; padding-left:16px; }
.timeline .t { position:relative; margin:0 0 12px; font-size:14px; }
.timeline .t::before { content:''; position:absolute; left:-23px; top:5px; width:10px; height:10px; border-radius:50%;
  background:#4F46E5; box-shadow:0 0 0 3px #EEF2FF; }
.answer { border-left:4px solid #4F46E5; background:#F8FAFC; padding:14px 18px; border-radius:0 12px 12px 0; font-size:1.02rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:12px 16px; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:500; }
div[data-testid="stTabs"] button { font-weight:500; }
</style>
"""

STEPS = [("Upload", "📥"), ("Quality", "🩺"), ("Clean", "🧹"), ("Explore", "🔎"), ("Insights", "🧠"), ("Export", "📦")]


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown("<div class='brand'><div class='logo'>IA</div><div><b>Intelligent Analytics</b><br>"
                        "<span>Guided &amp; explainable</span></div></div>", unsafe_allow_html=True)


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


def stepper(current: int, has_data: bool, cleaned: bool):
    out = []
    for i, (label, icon) in enumerate(STEPS):
        done = (i == 0 and has_data) or (i == 2 and cleaned) or (has_data and i < current and i != 2)
        cls = "now" if i == current else "done" if done else ("lock" if not has_data and i > 0 else "")
        out.append(f"<div class='step {cls}'><i>{'✓' if done and i != current else i + 1}</i>{icon} {label}</div>")
    st.markdown(f"<div class='stepper'>{''.join(out)}</div>", unsafe_allow_html=True)


def card(icon: str, title: str, text: str):
    st.markdown(f"<div class='card'><div class='ic'>{icon}</div><h4>{html.escape(title)}</h4>"
                f"<p>{html.escape(text)}</p></div>", unsafe_allow_html=True)


def kpi(label: str, value, sub: str = ""):
    st.markdown(f"<div class='kpi'><span>{html.escape(label)}</span><b>{html.escape(str(value))}</b>"
                f"<small>{html.escape(sub)}</small></div>", unsafe_allow_html=True)


def chip(text: str, kind: str = "mute") -> str:
    return f"<span class='chip {kind}'>{html.escape(text)}</span>"


def chips(items: list[tuple[str, str]]):
    st.markdown("".join(chip(t, k) for t, k in items), unsafe_allow_html=True)


def gauge(score: float, title: str = "Quality score", height: int = 210):
    color = GOOD if score >= 80 else WARN if score >= 55 else BAD
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 32, "color": "#0F172A"}},
        title={"text": title, "font": {"size": 14, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "showticklabels": False},
               "bar": {"color": color, "thickness": 0.28}, "bgcolor": "#F1F5F9", "borderwidth": 0}))
    fig.update_layout(height=height, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
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


def empty_state(icon: str, title: str, text: str):
    st.markdown(f"<div class='card' style='text-align:center;padding:42px'><div style='font-size:40px'>{icon}</div>"
                f"<h4 style='margin-top:6px'>{html.escape(title)}</h4><p>{html.escape(text)}</p></div>", unsafe_allow_html=True)
