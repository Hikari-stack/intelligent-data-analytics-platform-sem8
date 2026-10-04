"""Exploratory data analysis: stats, insights, guardrail warnings, charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

from core.theme import DIVERGING, ACCENT
from core.utils import is_id_like


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if not pd.api.types.is_bool_dtype(df[c]) and not is_id_like(df[c], c)]


def datetime_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="datetime").columns)


def categorical_cols(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    skip = set(numeric_cols(df)) | set(datetime_cols(df))   # computed once, not once per column
    cols = []
    for c in df.columns:
        if c in skip or is_id_like(df[c], c):
            continue
        if df[c].nunique(dropna=True) <= max_unique:
            cols.append(c)
    return cols


def summary_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_cols(df)
    return df[cols].describe().T.round(3) if cols else pd.DataFrame()


def summary_categorical(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in categorical_cols(df):
        s = df[c].dropna()
        if s.empty:
            continue
        top = s.value_counts().index[0]
        rows.append({"column": c, "unique": s.nunique(), "top_value": top,
                     "top_share_%": round(float((s == top).mean() * 100), 1)})
    return pd.DataFrame(rows)


def boolean_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() == 2]


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of number columns, plus True/False columns treated as 1/0."""
    data = df[numeric_cols(df)].copy()
    for c in boolean_cols(df):
        data[c] = df[c].map(lambda v: float(v) if pd.notna(v) else np.nan)
    return data.corr() if data.shape[1] >= 2 else pd.DataFrame()


def _strength_word(r: float) -> str:
    r = abs(r)
    return "very strongly" if r >= 0.8 else "strongly" if r >= 0.6 else "moderately"


def generate_insights(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Rule-based plain-language findings, strongest first."""
    found: list[tuple[float, str]] = []

    corr = correlation(df)
    cols, vals = list(corr.columns), corr.values
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            r = vals[i][j]
            if pd.notna(r) and abs(r) >= 0.5:
                b = cols[j]
                direction = "positively" if r > 0 else "negatively"
                found.append((abs(r), f"{a} and {b} are {_strength_word(r)} {direction} "
                                      f"correlated (r = {r:.2f})."))

    for c in numeric_cols(df):
        x = df[c].dropna()
        if len(x) > 8 and x.nunique() > 2:
            sk = x.skew()
            if abs(sk) > 1:
                found.append((0.5 + min(abs(sk), 3) / 10,
                              f"{c} is highly skewed (skew = {sk:.2f}); the median "
                              f"({x.median():,.2f}) describes a typical value better than the "
                              f"mean ({x.mean():,.2f})."))

    for c in categorical_cols(df):
        s = df[c].dropna()
        if len(s) and s.nunique() > 1:
            top = s.value_counts()
            share = top.iloc[0] / len(s)
            if share >= 0.5:
                found.append((share, f"{c}: '{top.index[0]}' makes up {share:.0%} of all rows."))

    for c in df.columns:
        m = df[c].isna().mean()
        if m >= 0.05:
            found.append((0.4 + m / 10, f"{c} is {m:.0%} missing."))

    found.extend(_association_insights(df))
    found.sort(key=lambda t: t[0], reverse=True)
    return [text for _, text in found[:top_n]]


def guardrails(df: pd.DataFrame) -> list[str]:
    """Warnings that stop users drawing wrong conclusions."""
    w = []
    if len(df) < 30:
        w.append(f"Only {len(df)} rows: statistics from so few rows are unreliable.")
    if len(numeric_cols(df)) >= 2:
        w.append("Correlation shows that two columns move together, not that one causes the other.")
    heavy = [c for c in df.columns if df[c].isna().mean() > 0.3]
    if heavy:
        w.append(f"Columns with over 30% missing values ({', '.join(heavy)}) may give misleading results.")
    w.extend(_extra_guardrails(df))
    return w


def correlation_heatmap(df: pd.DataFrame):
    corr = correlation(df)
    if corr.empty:
        return None
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale=DIVERGING, title="Correlation matrix")
    return fig


def chart_for(df: pd.DataFrame, x: str, y: str | None = None):
    """Pick a sensible chart from the column types."""
    num, dt = numeric_cols(df), datetime_cols(df)

    if is_id_like(df[x], x) or (y is not None and is_id_like(df[y], y)):
        return None  # charts of identifier columns carry no information
    if y is not None and x in num and y in dt:
        x, y = y, x  # date vs number: put the date on the x axis

    if y is None or y == x:
        if x in num:
            return px.histogram(df, x=x, marginal="box", title=f"Distribution of {x}")
        if x in dt:
            counts = df.set_index(x).resample("MS").size().reset_index(name="rows")
            return px.line(counts, x=x, y="rows", markers=True, title=f"Rows per month by {x}")
        vc = df[x].value_counts().head(20).reset_index()
        vc.columns = [x, "count"]
        return px.bar(vc, x=x, y="count", title=f"Counts of {x}")

    if x in num and y in num:
        fig = px.scatter(df, x=x, y=y, opacity=0.6, title=f"{y} vs {x}")
        d = df[[x, y]].dropna()
        if len(d) > 2 and d[x].nunique() > 1:
            m, b = np.polyfit(d[x], d[y], 1)
            xs = np.array([d[x].min(), d[x].max()])
            fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="trend",
                                     line=dict(color="#EF4444")))
        return fig
    if x in dt and y in num:
        d = df[[x, y]].dropna().sort_values(x)
        return px.line(d, x=x, y=y, title=f"{y} over time")
    if x not in num and x not in dt and y in num:
        return px.box(df, x=x, y=y, title=f"{y} by {x}")
    if y not in num and y not in dt and x in num:
        return px.box(df, x=y, y=x, title=f"{x} by {y}")
    return None


# ---------------------------------------------------------------- Task 4.1 / 4.3 additions

def cramers_v(df: pd.DataFrame, a: str, b: str) -> float:
    """Strength of association between two category columns: 0 = none, 1 = perfect."""
    table = pd.crosstab(df[a], df[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def _association_insights(df: pd.DataFrame, min_v: float = 0.3, max_cols: int = 20) -> list[tuple[float, str]]:
    cols = [c for c in categorical_cols(df) if 2 <= df[c].nunique() <= 12][:max_cols]
    found = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cramers_v(df, a, b)
            if v >= min_v:
                word = "strongly" if v >= 0.5 else "moderately"
                found.append((v, f"{a} and {b} are {word} associated (Cram\u00e9r's V = {v:.2f})."))
    return found


def missing_matrix(df: pd.DataFrame, max_rows: int = 300):
    """Heatmap of where values are missing (dark = missing)."""
    if not df.isna().any().any():
        return None
    sample = df.sample(min(len(df), max_rows), random_state=0) if len(df) > max_rows else df
    fig = px.imshow(sample.isna().astype(int).values, aspect="auto",
                    labels=dict(x="column", y="row", color="missing"),
                    x=list(sample.columns), color_continuous_scale=["#EEF2FF", "#EF4444"],
                    title="Missing values (dark = missing)")
    fig.update_coloraxes(showscale=False)
    return fig


def _extra_guardrails(df: pd.DataFrame) -> list[str]:
    w = []
    n = len(df)
    for c in df.columns:
        s = df[c].dropna()
        if s.empty or is_id_like(df[c], c):
            continue
        if s.nunique() > 50 and s.nunique() < 0.9 * len(s) and not pd.api.types.is_numeric_dtype(s) \
                and not pd.api.types.is_datetime64_any_dtype(s):
            w.append(f"'{c}' has {s.nunique()} different values: too many groups to read in a chart.")
        top_share = s.value_counts(normalize=True).iloc[0]
        if s.nunique() > 1 and top_share > 0.95:
            w.append(f"'{c}' has the same value in over 95% of rows; it says very little.")
    if n >= 30:
        for c in categorical_cols(df)[:12]:
            small = df[c].value_counts()
            small = small[small < 5]
            if 0 < len(small) < df[c].nunique():
                w.append(f"'{c}': group '{small.index[0]}' has only {int(small.iloc[0])} rows; "
                         "do not compare it with much larger groups.")
                break
    return w
