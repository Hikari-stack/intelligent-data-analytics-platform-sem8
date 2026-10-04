"""Ask-your-data: answer plain-English questions with no external service (rule-based).

It understands questions such as
  "average sales by region"          "total profit in the East"
  "which category has the highest sales"   "top 5 regions by profit"
  "how many orders where region is south"  "correlation between sales and profit"
  "sales over time" / "monthly sales"      "distribution of customer age"
and always shows the pandas code it ran, so the answer can be checked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px

from core import eda
from core.timeseries import aggregate

AGG_WORDS = {"average": "mean", "avg": "mean", "mean": "mean", "typical": "median", "median": "median",
             "total": "sum", "sum": "sum", "overall": "sum", "combined": "sum",
             "maximum": "max", "max": "max", "minimum": "min", "min": "min"}
HIGH = ("highest", "most", "largest", "biggest", "best", "top", "greatest", "maximum", "max")
LOW = ("lowest", "least", "smallest", "worst", "bottom", "minimum", "min", "fewest")
EXAMPLES = ["average {num} by {cat}", "total {num} by {cat}", "which {cat} has the highest {num}",
            "top 3 {cat} by {num}", "how many rows where {cat} is {val}", "correlation between {num} and {num2}",
            "distribution of {num}", "monthly {num}"]


@dataclass
class Answer:
    text: str
    table: pd.DataFrame | None = None
    figure: object | None = None
    code: str = ""
    ok: bool = True
    suggestions: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(s))      # CustomerAge -> Customer Age
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("_", " ")).strip()


def _find_columns(q: str, df: pd.DataFrame) -> list[str]:
    """Columns named in the question, in order of appearance; longest names win."""
    nq = f" {_norm(q)} "
    hits = []
    for c in sorted(df.columns, key=lambda c: -len(_norm(c))):
        n = _norm(c)
        if not n:
            continue
        for form in {n, n + "s", n[:-1] if n.endswith("s") else n}:
            m = re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", nq)
            if m and not any(a <= m.start() < b for _, a, b in hits):
                hits.append((c, m.start(), m.end()))
                nq = nq[:m.start()] + " " * (m.end() - m.start()) + nq[m.end():]
                break
    return [c for c, a, _ in sorted(hits, key=lambda t: t[1])]


def _find_values(q: str, df: pd.DataFrame, exclude: set[str]) -> list[tuple[str, object]]:
    """(column, value) pairs for category values mentioned in the question (e.g. 'East')."""
    nq = f" {_norm(q)} "
    found = []
    for c in eda.categorical_cols(df):
        if c in exclude or pd.api.types.is_bool_dtype(df[c]):
            continue
        for v in df[c].dropna().unique():
            nv = _norm(v)
            if len(nv) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(nv)}(?![a-z0-9])", nq):
                found.append((c, v))
    return found


def _suggest(df: pd.DataFrame) -> list[str]:
    num, cat = eda.numeric_cols(df), eda.categorical_cols(df)
    ex = {"num": num[0] if num else "column", "num2": num[1] if len(num) > 1 else (num[0] if num else "column"),
          "cat": cat[0] if cat else "column",
          "val": str(df[cat[0]].dropna().iloc[0]) if cat and df[cat[0]].notna().any() else "value"}
    return [e.format(**ex) for e in EXAMPLES if (num or "{num}" not in e) and (cat or "{cat}" not in e)][:6]


def _fail(msg: str, df: pd.DataFrame) -> Answer:
    return Answer(msg, ok=False, suggestions=_suggest(df))


def ask(df: pd.DataFrame, question: str) -> Answer:
    q = question.strip()
    if not q:
        return _fail("Type a question about your data.", df)
    ql = _norm(q)
    words = set(ql.split())
    cols = _find_columns(q, df)
    num_set, dt_set = set(eda.numeric_cols(df)), set(eda.datetime_cols(df))
    nums = [c for c in cols if c in num_set]
    dts = [c for c in cols if c in dt_set]
    cats = [c for c in cols if c not in num_set and c not in dt_set]

    # ---- missing values
    if "missing" in words and cols:
        c = cols[0]
        n = int(df[c].isna().sum())
        return Answer(f"**{c}** has **{n:,}** missing values ({n / max(len(df), 1):.1%}).",
                      code=f"df[{c!r}].isna().sum()")

    # ---- row counts
    if re.search(r"\b(how many|number of|count)\b", ql) and not nums and not dts:
        pairs = _find_values(q, df, set())
        sub, code = df, "df"
        for c, v in pairs:
            sub = sub[sub[c] == v]
            code += f"[df[{c!r}] == {v!r}]"
        cond = " and ".join(f"{c} = {v}" for c, v in pairs)
        what = f"rows where {cond}" if pairs else "rows"
        return Answer(f"There are **{len(sub):,}** {what}.", code=f"len({code})")

    # ---- correlation
    if "correlation" in words or "correlated" in words or "relationship" in words or "related" in words:
        if len(nums) >= 2:
            a, b = nums[:2]
            r = df[a].corr(df[b])
            strength = eda._strength_word(r) if abs(r) >= 0.5 else "weakly"
            fig = eda.chart_for(df, a, b)
            return Answer(f"**{a}** and **{b}** are {strength} {'positively' if r > 0 else 'negatively'} "
                          f"correlated (r = {r:.2f}). This shows they move together, not that one causes the other.",
                          figure=fig, code=f"df[{a!r}].corr(df[{b!r}])")
        return _fail("Name two numeric columns, e.g. 'correlation between sales and profit'.", df)

    # ---- distribution
    if words & {"distribution", "histogram", "spread"} and cols:
        c = cols[0]
        fig = eda.chart_for(df, c)
        if fig is not None:
            return Answer(f"Here is the distribution of **{c}**.", figure=fig, code=f"px.histogram(df, x={c!r})")

    # ---- over time
    time_words = words & {"trend", "monthly", "weekly", "daily", "yearly", "quarterly", "time", "month", "year"}
    if (time_words or dts) and dt_set and ("over" in words or "trend" in words or "per" in words or time_words):
        d = dts[0] if dts else sorted(dt_set)[0]
        v = nums[0] if nums else None
        freq = ("Week" if words & {"weekly", "week"} else "Day" if words & {"daily", "day"} else
                "Quarter" if words & {"quarterly", "quarter"} else "Year" if words & {"yearly", "year", "annual"} else "Month")
        agg = next((AGG_WORDS[w] for w in ql.split() if w in AGG_WORDS), "sum" if v else "count")
        s = aggregate(df, d, v, freq, agg)
        fig = px.line(s.reset_index(), x=s.index.name or d, y=s.name, markers=True,
                      title=f"{agg.title()} of {v or 'rows'} per {freq.lower()}")
        best = s.idxmax()
        return Answer(f"{freq}ly {agg} of **{v or 'rows'}**. Highest in {best:%b %Y} ({s.max():,.2f}); "
                      f"lowest in {s.idxmin():%b %Y} ({s.min():,.2f}).", table=s.reset_index(), figure=fig,
                      code=f"df.set_index({d!r})[{v!r}].resample('{ {'Day':'D','Week':'W','Month':'MS','Quarter':'QS','Year':'YS'}[freq] }').{agg}()")

    # ---- group-by / aggregate
    if nums or (cats and re.search(r"\b(how many|count)\b", ql)):
        value = nums[0] if nums else None
        group = cats[0] if cats else None
        agg = next((AGG_WORDS[w] for w in ql.split() if w in AGG_WORDS), None)
        wants_high = any(w in ql.split() for w in HIGH)
        wants_low = any(w in ql.split() for w in LOW)
        n_top = next((int(m) for m in re.findall(r"\b(?:top|bottom|first)\s+(\d+)\b", ql)), None)
        pairs = [(c, v) for c, v in _find_values(q, df, {group} if group else set())]
        sub, filt_code, filt_txt = df, "df", ""
        for c, v in pairs:
            sub = sub[sub[c] == v]
            filt_code += f"[df[{c!r}] == {v!r}]"
        if pairs:
            filt_txt = " for " + " and ".join(f"{c} = {v}" for c, v in pairs)
        if sub.empty:
            return _fail("No rows match those filters.", df)

        if value and group:
            agg = agg or "sum"
            res = sub.groupby(group, dropna=True)[value].agg(agg).sort_values(ascending=wants_low and not wants_high)
            res = res.reset_index()
            shown = res.head(n_top) if n_top else res
            fig = px.bar(shown.head(25), x=group, y=value, title=f"{agg.title()} of {value} by {group}{filt_txt}")
            if wants_high or wants_low or n_top:
                top = res.iloc[0]
                word = "lowest" if wants_low and not wants_high else "highest"
                text = (f"**{top[group]}** has the {word} {agg} of {value} "
                        f"({top[value]:,.2f}){filt_txt}.")
            else:
                text = f"{agg.title()} of **{value}** by **{group}**{filt_txt}. Highest: {res.iloc[0][group]} ({res.iloc[0][value]:,.2f})."
            return Answer(text, table=shown, figure=fig,
                          code=f"{filt_code}.groupby({group!r})[{value!r}].{agg}().sort_values()")
        if value:
            agg = agg or ("mean" if "average" in ql or "typical" in ql else "sum")
            x = sub[value].astype(float)
            val = getattr(x, agg)()
            return Answer(f"The {agg} of **{value}**{filt_txt} is **{val:,.2f}** (from {int(x.notna().sum()):,} rows).",
                          code=f"{filt_code}[{value!r}].{agg}()")
        if group:
            vc = sub[group].value_counts().reset_index()
            vc.columns = [group, "count"]
            fig = px.bar(vc.head(25), x=group, y="count", title=f"Rows per {group}")
            return Answer(f"**{vc.iloc[0][group]}** is the most common {group} ({vc.iloc[0]['count']:,} rows).",
                          table=vc, figure=fig, code=f"df[{group!r}].value_counts()")

    if cats:   # a bare column name -> show its breakdown
        c = cats[0]
        vc = df[c].value_counts().reset_index()
        vc.columns = [c, "count"]
        return Answer(f"Breakdown of **{c}**.", table=vc, figure=px.bar(vc.head(25), x=c, y="count"),
                      code=f"df[{c!r}].value_counts()")
    return _fail("I could not match that to your columns. Mention a column name and what you want "
                 "(average, total, highest, top 5, correlation, over time...).", df)
