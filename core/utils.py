"""Small helpers shared by profiling and cleaning."""
from __future__ import annotations

import re

import pandas as pd

# Strings that really mean "missing"
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing"}

DATE_RE = re.compile(
    r"^\s*(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})"
)


def is_text_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def null_token_mask(s: pd.Series) -> pd.Series:
    """True where a value is a string like 'N/A', 'null', '-' or blank."""
    return s.map(lambda v: isinstance(v, str) and v.strip().lower() in NULL_TOKENS).astype(bool)


def to_numeric_loose(s: pd.Series) -> pd.Series:
    """'1,200.50', '$300', '12%' -> numbers. Unparseable values become NaN."""
    base = s.astype(object).where(~null_token_mask(s), None)

    def conv(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        if isinstance(v, (int, float)):
            return float(v)
        cleaned = re.sub(r"[,\s$₹€£%]", "", str(v))
        try:
            return float(cleaned)
        except ValueError:
            return float("nan")

    return base.map(conv).astype(float)


def to_datetime_loose(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    vals = s.astype(object).where(~null_token_mask(s), None)
    return pd.to_datetime(vals, errors="coerce", format="mixed", dayfirst=dayfirst)


def looks_numeric(s: pd.Series, threshold: float = 0.8) -> bool:
    """Text column where most values are really numbers (e.g. '1,200')."""
    if not is_text_dtype(s):
        return False
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return False
    return to_numeric_loose(vals).notna().mean() >= threshold


_DMY_RE = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}")


def guess_dayfirst(s: pd.Series) -> bool:
    """Day-first (31/01/2024) or month-first (01/31/2024)? Decided from values that can only be
    one or the other; when every value is ambiguous (e.g. 02/03/2024) assume day-first."""
    day_first_seen = month_first_seen = False
    for v in s.dropna().astype(str):
        m = _DMY_RE.match(v)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        day_first_seen |= a > 12
        month_first_seen |= b > 12
    return not (month_first_seen and not day_first_seen)


def looks_datetime(s: pd.Series, threshold: float = 0.8) -> bool:
    """Text column where most values are dates in some format."""
    if not is_text_dtype(s):
        return False
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return False
    matches = vals.map(lambda v: bool(DATE_RE.match(str(v)))).mean()
    if matches < threshold:
        return False
    return to_datetime_loose(vals).notna().mean() >= threshold


def tidy_text(v):
    """Trim and collapse inner whitespace; leave non-strings alone."""
    return re.sub(r"\s+", " ", v.strip()) if isinstance(v, str) else v


_ID_NAME_RE = re.compile(r"(?:^|[_\s\-])id(?:$|[_\s\-])|[a-z]ID$|^index$|_no$|_number$", re.IGNORECASE)


def is_id_like(s: pd.Series, name: str | None = None) -> bool:
    """True for identifier columns (OrderID, customer_id, row numbers).

    These are whole numbers, never missing, all different, and either named like an ID or a
    plain 1,2,3... sequence. Statistics, correlations and outlier checks on them are meaningless.
    """
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    n = len(s)
    if n < 10 or s.isna().any() or s.nunique() != n:
        return False
    if not (s % 1 == 0).all():
        return False
    named = bool(_ID_NAME_RE.search(str(name if name is not None else s.name or "")))
    sequential = pd.api.types.is_integer_dtype(s) and (s.max() - s.min() + 1) == n
    return named or sequential


TRUE_WORDS = {"yes", "y", "true", "t", "1", "1.0"}
FALSE_WORDS = {"no", "n", "false", "f", "0", "0.0"}


def to_boolean_loose(s: pd.Series) -> pd.Series:
    """yes/no, y/n, true/false, 1/0 -> True/False. Anything else becomes missing."""

    def conv(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, bool):
            return v
        t = str(v).strip().lower()
        if t in TRUE_WORDS:
            return True
        if t in FALSE_WORDS:
            return False
        return None

    base = s.astype(object).where(~null_token_mask(s), None)
    return base.map(conv).astype("boolean")


def looks_boolean(s: pd.Series) -> bool:
    """Text column whose values are all yes/no style answers (at least one is a word)."""
    if not is_text_dtype(s):
        return False
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return False
    words = vals.map(lambda v: str(v).strip().lower())
    if not words.isin(TRUE_WORDS | FALSE_WORDS).all():
        return False
    return bool(words.str.contains("[a-z]").any())


def numeric_share(s: pd.Series) -> float:
    """Fraction of a text column's filled values that are really numbers."""
    if not is_text_dtype(s):
        return 0.0
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return 0.0
    return float(to_numeric_loose(vals).notna().mean())


def non_numeric_distinct(s: pd.Series) -> int:
    """How many different non-number values a text column contains."""
    if not is_text_dtype(s):
        return 0
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return 0
    words = vals[to_numeric_loose(vals).isna()]
    return int(words.nunique())
