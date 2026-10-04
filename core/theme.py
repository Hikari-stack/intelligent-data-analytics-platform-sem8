"""One colour palette and one Plotly template shared by every chart in the app."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

ACCENT = "#1E5A78"      # petrol blue
GOOD, WARN, BAD = "#3F7D58", "#B7791F", "#B3412F"
INK, MUTED, LINE = "#1F2933", "#5F6B76", "#DAD6CE"
PAPER = "#FBFAF7"
PALETTE = ["#1E5A78", "#C1692F", "#4F8A67", "#8C6A9E", "#C9A227", "#6B7A86", "#A8402F", "#3A8FA3"]
DIVERGING = [[0, BAD], [0.5, PAPER], [1, ACCENT]]
SEQUENTIAL = [[0, "#E4EEF3"], [1, ACCENT]]

_FONT = "IBM Plex Sans, -apple-system, Segoe UI, Roboto, sans-serif"

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family=_FONT, size=13, color=INK),
    title=dict(font=dict(size=15, color=INK), x=0.0, xanchor="left"),
    colorway=PALETTE,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=56, b=40),
    xaxis=dict(gridcolor="#ECE9E2", linecolor=LINE, zeroline=False, ticks="outside", tickcolor=LINE),
    yaxis=dict(gridcolor="#ECE9E2", linecolor=LINE, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(font=dict(family=_FONT), bgcolor="#FFFFFF", bordercolor=LINE),
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
