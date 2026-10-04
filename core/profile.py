"""Data quality profiling and fix suggestions."""
from __future__ import annotations

import pandas as pd

from core.utils import (
    guess_dayfirst,
    is_id_like,
    is_text_dtype,
    looks_boolean,
    looks_datetime,
    looks_numeric,
    non_numeric_distinct,
    numeric_share,
    null_token_mask,
    tidy_text,
)

MAX_CATEGORY_UNIQUE = 50  # only look for spelling variants in short category lists


def _kind(s: pd.Series, numeric_text: bool, datetime_text: bool, id_like: bool = False,
          boolean_text: bool = False) -> str:
    if id_like:
        return "id"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if boolean_text:
        return "boolean_text"
    if numeric_text:
        return "numeric_text"
    if datetime_text:
        return "datetime_text"
    n = s.notna().sum()
    if n and s.nunique(dropna=True) / n > 0.9 and s.nunique(dropna=True) > 20:
        return "text"
    return "categorical"


def _variants(s: pd.Series) -> list[list[str]]:
    """Groups of spellings that differ only by case/spacing, e.g. ['West', 'west ']."""
    vals = s.dropna()
    if vals.empty or vals.nunique() > MAX_CATEGORY_UNIQUE:
        return []
    groups: dict[str, set] = {}
    for v in vals.unique():
        key = str(tidy_text(v)).lower()
        groups.setdefault(key, set()).add(v)
    return [sorted(map(str, g)) for g in groups.values() if len(g) > 1]


def profile_column(s: pd.Series, name: str | None = None) -> dict:
    n = len(s)
    text = is_text_dtype(s)
    tokens = int(null_token_mask(s).sum()) if text else 0
    num_text = looks_numeric(s)
    dt_text = looks_datetime(s) if not num_text else False
    bool_text = looks_boolean(s)
    kind = _kind(s, num_text, dt_text, is_id_like(s, name), bool_text)
    share = numeric_share(s)
    mixed = bool(text and kind in ("categorical", "text") and 0.1 <= share < 0.8
                 and non_numeric_distinct(s) <= 20)  # many different words = a code column, not a typo

    info = {
        "dtype": str(s.dtype),
        "kind": kind,
        "missing": int(s.isna().sum()),
        "missing_pct": round(float(s.isna().mean() * 100), 2) if n else 0.0,
        "null_tokens": tokens,
        "unique": int(s.nunique(dropna=True)),
        "constant": bool(s.nunique(dropna=True) == 1),
        "mixed": mixed,
        "outliers": 0,
        "outlier_pct": 0.0,
        "skew": None,
        "integer_valued": False,
        "variants": [],
        "dayfirst": guess_dayfirst(s) if kind == "datetime_text" else True,
    }

    if kind == "numeric":
        x = s.dropna().astype(float)
        info["integer_valued"] = bool(len(x) and (x % 1 == 0).all())
        if len(x) >= 4:
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
                info["outliers"] = int(out)
                info["outlier_pct"] = round(float(out / len(x) * 100), 2)
            info["skew"] = round(float(x.skew()), 3) if x.nunique() > 2 else 0.0
    elif kind == "categorical":
        info["variants"] = _variants(s)

    return info


def profile(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    cols = {c: profile_column(df[c], c) for c in df.columns}

    dup = int(df.duplicated().sum())
    total = max(n_rows * n_cols, 1)
    missing_cells = int(df.isna().sum().sum())
    missing_pct = missing_cells / total * 100
    dup_pct = dup / max(n_rows, 1) * 100

    flagged = sum(
        1
        for i in cols.values()
        if i["kind"] in ("numeric_text", "datetime_text", "boolean_text")
        or i["mixed"]
        or i["variants"]
        or i["null_tokens"]
        or i["constant"]
        or i["outlier_pct"] > 5
    )
    score = 100 - min(40, missing_pct * 1.5) - min(20, dup_pct * 2) - min(40, 40 * flagged / max(n_cols, 1))

    return {
        "overview": {
            "rows": n_rows,
            "columns": n_cols,
            "duplicate_rows": dup,
            "missing_cells": missing_cells,
            "missing_pct": round(missing_pct, 2),
            "flagged_columns": flagged,
            "quality_score": round(max(score, 0), 1),
        },
        "columns": cols,
    }


def suggest_fixes(prof: dict) -> list[dict]:
    """Turn findings into suggestions. Each one is a ready-to-apply pipeline step with a reason.

    Order matters: structural fixes first, imputation last. A column that still has a pending
    conversion/cleanup never gets an imputation suggestion until that is done.
    """
    steps: list[dict] = []
    ov, cols = prof["overview"], prof["columns"]

    if ov["duplicate_rows"] > 0:
        steps.append({"op": "drop_duplicates",
                      "reason": f"{ov['duplicate_rows']} fully duplicated rows found."})

    if any(i["null_tokens"] for i in cols.values()):
        total = sum(i["null_tokens"] for i in cols.values())
        steps.append({"op": "normalize_missing",
                      "reason": f"{total} cells contain placeholders like 'N/A', 'null' or '-'. "
                                "Convert them to real missing values."})

    for c, i in cols.items():
        if i["kind"] == "numeric_text":
            steps.append({"op": "convert_numeric", "col": c,
                          "reason": f"'{c}' is stored as text but looks numeric (e.g. '1,200')."})
        elif i["kind"] == "boolean_text":
            steps.append({"op": "convert_boolean", "col": c,
                          "reason": f"'{c}' holds yes/no style answers stored as text; "
                                    "convert them to True/False."})
        elif i["kind"] == "datetime_text":
            steps.append({"op": "convert_datetime", "col": c, "dayfirst": i["dayfirst"],
                          "reason": f"'{c}' is stored as text but looks like dates "
                                    f"({'day-first' if i['dayfirst'] else 'month-first'} order detected)."})
        elif i["variants"]:
            ex = i["variants"][0]
            steps.append({"op": "standardize_categories", "col": c,
                          "reason": f"'{c}' has inconsistent spellings, e.g. {ex}."})

    for c, i in cols.items():
        if i["constant"]:
            steps.append({"op": "drop_column", "col": c,
                          "reason": f"'{c}' has the same value in every row, so it adds no information."})

    for c, i in cols.items():
        pending = (i["kind"] in ("numeric_text", "datetime_text", "boolean_text") or i["variants"]
                   or i["null_tokens"] or i["constant"])
        if i["missing"] == 0 or pending:
            continue
        if i["missing_pct"] > 60:
            steps.append({"op": "drop_column", "col": c,
                          "reason": f"'{c}' is {i['missing_pct']}% missing; too little data to be useful."})
        elif i["kind"] == "numeric":
            skewed = i["skew"] is not None and abs(i["skew"]) > 1
            if i["integer_valued"]:
                method = "median"
                why = "a whole-number column, so the median keeps a valid value (no 4.9 items)"
            elif skewed:
                method = "median"
                why = f"skewed (skew={i['skew']}), so the median is more representative"
            else:
                method = "mean"
                why = "roughly symmetric, so the mean is a fair fill value"
            note = (" Note: filling many cells with one value creates a spike in charts; "
                    "check the Before/After view." if i["missing_pct"] >= 5 else "")
            steps.append({"op": "impute", "col": c, "method": method,
                          "reason": f"'{c}' has {i['missing_pct']}% missing and is {why}.{note}"})
        elif i["kind"] in ("categorical", "boolean", "text"):
            steps.append({"op": "impute", "col": c, "method": "mode",
                          "reason": f"'{c}' has {i['missing_pct']}% missing; fill with the most common value."})

    for c, i in cols.items():
        if i["kind"] == "numeric" and i["outlier_pct"] > 1:
            steps.append({"op": "cap_outliers", "col": c,
                          "reason": f"'{c}' has {i['outliers']} outliers ({i['outlier_pct']}%). "
                                    "Cap them at the IQR limits so they don't distort results."})
    return steps


def describe_step(step: dict) -> str:
    """Short human label for a step."""
    op, col = step["op"], step.get("col")
    labels = {
        "drop_duplicates": "Remove duplicate rows",
        "normalize_missing": "Convert placeholders to missing values",
        "convert_numeric": f"Convert '{col}' to numbers",
        "convert_datetime": f"Convert '{col}' to dates",
        "convert_boolean": f"Convert '{col}' to True/False",
        "drop_rows_missing": f"Drop rows with more than {int(step.get('threshold', 0.5) * 100)}% missing values",
        "replace_values": f"Replace values in '{col}'",
        "rename_column": f"Rename '{col}' to '{step.get('new_name')}'",
        "standardize_categories": f"Standardize spellings in '{col}'",
        "drop_column": f"Drop column '{col}'",
        "impute": f"Fill missing '{col}' with {step.get('method')}",
        "cap_outliers": f"Cap outliers in '{col}'",
        "filter_rows": f"Keep rows where '{col}' {step.get('operator')} {step.get('value', '')}".rstrip(),
        "find_replace": f"In '{col}', replace '{step.get('find')}' with '{step.get('replace', '')}'",
        "drop_outlier_rows": f"Remove outlier rows in '{col}' ({step.get('method', 'iqr')})",
    }
    if op == "impute" and step.get("method") == "knn":
        return f"Fill missing '{col}' from the {step.get('k', 5)} most similar rows (KNN)"
    return labels.get(op, op)
