"""Time-series helpers: aggregate by period, trend, seasonality, plain-language summary."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.theme import ACCENT, MUTED, PALETTE

FREQS = {"Day": "D", "Week": "W", "Month": "MS", "Quarter": "QS", "Year": "YS"}
AGGS = ("sum", "mean", "median", "count", "max", "min")


def aggregate(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str = "Month",
              agg: str = "sum") -> pd.Series:
    if freq not in FREQS or agg not in AGGS:
        raise ValueError("Unknown frequency or aggregation")
    d = df[[date_col] + ([value_col] if value_col else [])].dropna(subset=[date_col]).set_index(date_col).sort_index()
    if value_col is None or agg == "count":
        s = d.resample(FREQS[freq]).size()
    else:
        s = d[value_col].resample(FREQS[freq]).agg(agg)
        if agg in ("sum", "count"):
            s = s.fillna(0)
    return s.rename(value_col or "rows")


def analyse(series: pd.Series, window: int = 3) -> dict:
    """Moving average, linear trend and seasonal pattern (month-of-year) with a short summary."""
    s = series.dropna()
    res = {"series": series, "ma": series.rolling(window, min_periods=1).mean(), "summary": [], "seasonal": None}
    if len(s) < 4:
        res["summary"].append("Too few periods to describe a trend.")
        return res
    import numpy as np
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1)
    mean = s.mean() or 1
    pct = slope / abs(mean) * 100
    direction = "upward" if pct > 1 else "downward" if pct < -1 else "flat"
    res["summary"].append(f"Overall trend is {direction} ({pct:+.1f}% of the average per period).")
    res["summary"].append(f"Highest period: {s.idxmax():%b %Y} ({s.max():,.2f}); lowest: {s.idxmin():%b %Y} ({s.min():,.2f}).")
    last, prev = s.iloc[-1], s.iloc[-2]
    if prev:
        res["summary"].append(f"Latest period changed {((last - prev) / abs(prev)) * 100:+.1f}% versus the one before.")
    if len(s) >= 24 and hasattr(s.index, "month"):
        seas = s.groupby(s.index.month).mean()
        res["seasonal"] = seas
        res["summary"].append(f"Strongest month on average: {pd.Timestamp(2000, int(seas.idxmax()), 1):%B}; "
                              f"weakest: {pd.Timestamp(2000, int(seas.idxmin()), 1):%B}.")
    return res


def figure(res: dict, title: str = ""):
    s = res["series"]
    seas = res["seasonal"]
    rows = 2 if seas is not None else 1
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.18,
                        subplot_titles=(title, "Average by calendar month") if rows == 2 else (title,))
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=s.name,
                             line=dict(color=ACCENT, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=res["ma"].values, mode="lines", name="moving average",
                             line=dict(color=PALETTE[3], width=2, dash="dash")), row=1, col=1)
    if seas is not None:
        fig.add_trace(go.Bar(x=[pd.Timestamp(2000, int(m), 1).strftime("%b") for m in seas.index],
                             y=seas.values, name="seasonal", marker_color=PALETTE[1]), row=2, col=1)
    fig.update_layout(height=340 * rows, showlegend=True, template="idap")
    return fig
