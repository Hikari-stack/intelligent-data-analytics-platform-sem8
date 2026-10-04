"""One colour palette and one Plotly template shared by every chart in the app."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

ACCENT = "#4F46E5"      # indigo
GOOD, WARN, BAD = "#10B981", "#F59E0B", "#EF4444"
INK, MUTED, LINE = "#0F172A", "#64748B", "#E2E8F0"
PALETTE = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#64748B", "#EF4444"]
DIVERGING = [[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#4F46E5"]]
SEQUENTIAL = [[0, "#EEF2FF"], [1, "#4F46E5"]]

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=13, color=INK),
    title=dict(font=dict(size=16, color=INK), x=0.0, xanchor="left"),
    colorway=PALETTE,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=56, b=40),
    xaxis=dict(gridcolor=LINE, linecolor=LINE, zeroline=False),
    yaxis=dict(gridcolor=LINE, linecolor=LINE, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(font=dict(family="Inter, sans-serif")),
)
pio.templates["idap"] = _template
pio.templates.default = "idap"


def style(fig, height: int | None = 380):
    """Final touches applied to any figure just before it is shown."""
    if fig is None:
        return None
    fig.update_layout(template="idap")
    if height:
        fig.update_layout(height=height)
    return fig
