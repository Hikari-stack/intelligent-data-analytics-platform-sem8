"""Cleaning operations and the replayable pipeline log.

A pipeline is just a list of dicts, e.g. {"op": "impute", "col": "Age", "method": "median"}.
Because it is plain data, it can be saved as JSON, undone, and replayed on a new file.
"""
from __future__ import annotations

import math

import pandas as pd

from core.utils import (
    null_token_mask,
    tidy_text,
    to_boolean_loose,
    to_datetime_loose,
    to_numeric_loose,
)

COLUMN_OPS = {"convert_numeric", "convert_datetime", "convert_boolean", "standardize_categories",
              "impute", "cap_outliers", "replace_values", "rename_column", "filter_rows",
              "find_replace", "drop_outlier_rows"}
FILTER_OPERATORS = ("==", "!=", ">", ">=", "<", "<=", "contains", "not contains", "is missing",
                    "is not missing")
ALL_OPS = COLUMN_OPS | {"drop_duplicates", "normalize_missing", "drop_column", "drop_rows_missing"}


def _require_col(df: pd.DataFrame, col: str | None) -> str:
    if col is None or col not in df.columns:
        raise ValueError(f"Column '{col}' not found in this dataset.")
    return col


def _require_numeric(s: pd.Series, col: str, what: str) -> pd.Series:
    """Return the column as floats, or raise a message the user can act on."""
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"Cannot {what} '{col}' because it is not a numeric column. "
                         "Convert it to numbers first, or use a different method.")
    return s.astype(float)


def outlier_mask(x: pd.Series, method: str = "iqr") -> pd.Series:
    """True where a value is an outlier. method: 'iqr' (1.5 x IQR) or 'zscore' (|z| > 3)."""
    if method == "zscore":
        sd = x.std()
        if not sd or pd.isna(sd):
            return pd.Series(False, index=x.index)
        return ((x - x.mean()).abs() / sd) > 3
    if method != "iqr":
        raise ValueError(f"Unknown outlier method: {method!r}")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)


def _knn_fill(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Fill gaps in `col` from the k most similar rows, judged on the other numeric columns."""
    from sklearn.impute import KNNImputer
    from core.utils import is_id_like
    feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
             and not pd.api.types.is_bool_dtype(df[c]) and not is_id_like(df[c], c)]
    if col not in feats:
        feats.append(col)
    data = df[feats].astype(float)
    if data[col].notna().sum() < 2:
        return df[col]
    std = data.std().replace(0, 1)
    scaled = (data - data.mean()) / std               # so no column dominates the distance
    filled = KNNImputer(n_neighbors=max(1, k)).fit_transform(scaled)
    res = pd.Series(filled[:, feats.index(col)] * std[col] + data[col].mean(), index=df.index)
    known = data[col].dropna()
    if (known % 1 == 0).all():
        res = res.round()
    return df[col].where(df[col].notna(), res)


def _filter_rows(df: pd.DataFrame, col: str, op: str | None, value) -> pd.DataFrame:
    """Keep only the rows where `col <op> value` is true."""
    if op not in FILTER_OPERATORS:
        raise ValueError(f"Unknown filter operator: {op!r}")
    s = df[col]
    if op == "is missing":
        keep = s.isna()
    elif op == "is not missing":
        keep = s.notna()
    elif op in ("contains", "not contains"):
        hit = s.astype(str).str.contains(str(value), case=False, regex=False) & s.notna()
        keep = hit if op == "contains" else ~hit
    else:
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{col}' is numeric, so '{value}' must be a number.")
        elif pd.api.types.is_datetime64_any_dtype(s):
            v = pd.to_datetime(value, errors="coerce")
            if pd.isna(v):
                raise ValueError(f"'{value}' is not a valid date.")
        elif pd.api.types.is_bool_dtype(s):
            v = str(value).strip().lower() in ("true", "1", "yes")
        else:
            v = str(value)
            s = s.astype(object).where(s.isna(), s.astype(str))
        keep = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
        keep = keep.fillna(False)
    if int(keep.sum()) == 0:
        raise ValueError("That filter would remove every row.")
    return df[keep].reset_index(drop=True)


def _pick_spelling(group: pd.Series) -> str:
    """Most common spelling; on ties prefer 'Title Case' over ALL CAPS or all lowercase."""
    counts = group.value_counts()
    return sorted(counts.index, key=lambda v: (-counts[v], not v.istitle(), v.isupper(), v.islower(), v))[0]


def apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Apply one step and return a new DataFrame (the input is never modified)."""
    op = step.get("op")
    if op not in ALL_OPS:
        raise ValueError(f"Unknown operation: {op!r}")
    out = df.copy()

    if op == "drop_duplicates":
        return out.drop_duplicates().reset_index(drop=True)

    if op == "normalize_missing":
        for c in out.columns:
            m = null_token_mask(out[c])
            if m.any():
                out[c] = out[c].astype(object).where(~m, None)
        return out

    if op == "drop_column":
        return out.drop(columns=[step["col"]], errors="ignore")

    if op == "drop_rows_missing":
        threshold = float(step.get("threshold", 0.5))
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than 0 and at most 1")
        keep = out.isna().mean(axis=1) <= threshold
        return out[keep].reset_index(drop=True)

    col = _require_col(out, step.get("col"))

    if op == "convert_numeric":
        out[col] = to_numeric_loose(out[col])

    elif op == "convert_boolean":
        out[col] = to_boolean_loose(out[col])

    elif op == "replace_values":
        mapping = step.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("replace_values needs a non-empty 'mapping' dictionary")
        out[col] = out[col].map(lambda v: mapping.get(v, v))

    elif op == "rename_column":
        new_name = str(step.get("new_name", "")).strip()
        if not new_name:
            raise ValueError("rename_column needs a 'new_name'")
        if new_name != col and new_name in out.columns:
            raise ValueError(f"A column called '{new_name}' already exists")
        out = out.rename(columns={col: new_name})

    elif op == "convert_datetime":
        out[col] = to_datetime_loose(out[col], dayfirst=step.get("dayfirst", True))

    elif op == "standardize_categories":
        s = out[col]
        tidy = s.map(tidy_text)
        key = tidy.map(lambda v: v.lower() if isinstance(v, str) else v)
        canon = tidy.groupby(key).agg(_pick_spelling)
        out[col] = key.map(canon)

    elif op == "impute":
        method = step.get("method", "median")
        s = out[col]
        if method in ("mean", "median"):
            num = _require_numeric(s, col, f"fill with the {method} for")
            fill = num.mean() if method == "mean" else num.median()
            known = num.dropna()
            if len(known) and pd.notna(fill) and (known % 1 == 0).all():
                fill = round(fill)  # whole-number column: fill with a whole number
        elif method == "mode":
            m = s.mode()
            fill = m.iat[0] if not m.empty else None
        elif method == "constant":
            fill = step.get("value")
        elif method == "knn":
            _require_numeric(s, col, "fill with nearest neighbours for")
            out[col] = _knn_fill(out, col, int(step.get("k", 5)))
            return out
        else:
            raise ValueError(f"Unknown impute method: {method!r}")
        if fill is not None and not (isinstance(fill, float) and pd.isna(fill)):
            out[col] = s.fillna(fill)

    elif op == "filter_rows":
        out = _filter_rows(out, col, step.get("operator"), step.get("value"))

    elif op == "find_replace":
        find = step.get("find")
        if find is None or str(find) == "":
            raise ValueError("find_replace needs a non-empty 'find' text")
        repl = str(step.get("replace", ""))
        out[col] = out[col].map(lambda v: v.replace(str(find), repl) if isinstance(v, str) else v)

    elif op == "drop_outlier_rows":
        x = _require_numeric(out[col], col, "find outlier rows in")
        out = out[~outlier_mask(x, step.get("method", "iqr"))].reset_index(drop=True)

    elif op == "cap_outliers":
        x = _require_numeric(out[col], col, "cap outliers in")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        known = x.dropna()
        if len(known) and (known % 1 == 0).all():   # whole-number column: keep limits whole
            low, high = math.ceil(low), math.floor(high)
        out[col] = x.clip(low, high)

    return out


def replay(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    """Re-run a list of steps from scratch on `df`. Undo = replay(df, steps[:-1])."""
    out = df.copy()
    for step in steps:
        out = apply_step(out, step)
    return out
