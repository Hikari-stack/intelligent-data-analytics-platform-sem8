"""Task 5.3: strange files must never crash the app. They either work or give a clear message."""
import io

import numpy as np
import pandas as pd
import pytest

from core.clean import replay
from core.eda import generate_insights, guardrails
from core.ingest import IngestError, load_file
from core.profile import profile, suggest_fixes


def full_pipeline(df):
    """profile -> apply every suggestion until none remain -> EDA. Must not raise."""
    steps = []
    for _ in range(10):
        sugg = suggest_fixes(profile(replay(df, steps)))
        if not sugg:
            break
        steps += sugg
    cleaned = replay(df, steps)
    generate_insights(cleaned)
    guardrails(cleaned)
    return cleaned


def test_empty_file():
    with pytest.raises(IngestError):
        load_file(io.BytesIO(b""), "a.csv")


def test_header_only_file():
    with pytest.raises(IngestError):
        load_file(io.BytesIO(b"a,b,c\n"), "a.csv")


def test_one_row_file():
    df = load_file(io.BytesIO(b"a,b\n1,2\n"), "a.csv")
    full_pipeline(df)


def test_single_column_file():
    df = load_file(io.BytesIO(b"score\n1\n2\n3\n4\n"), "a.csv")
    assert list(df.columns) == ["score"]
    full_pipeline(df)


def test_200_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 200)), columns=[f"c{i}" for i in range(200)])
    assert full_pipeline(df).shape[1] == 200


def test_all_missing_column_is_dropped():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "empty": [None] * 4})
    assert "empty" not in full_pipeline(df).columns


def test_latin1_encoding():
    raw = "name,city\nJos\u00e9,Z\u00fcrich\nAna,Madrid\n".encode("latin-1")
    df = load_file(io.BytesIO(raw), "a.csv")
    assert "Z" in df["city"].iloc[0]


def test_duplicate_column_names():
    df = load_file(io.BytesIO(b"a,a,b\n1,2,3\n4,5,6\n"), "a.csv")
    assert len(set(df.columns)) == 3
    full_pipeline(df)


def test_only_text_columns():
    df = pd.DataFrame({"x": list("abcdefghij"), "y": list("klmnopqrst")})
    full_pipeline(df)


def test_every_value_identical():
    df = pd.DataFrame({"a": [5] * 20, "b": ["z"] * 20})
    full_pipeline(df)


def test_columns_with_spaces_and_symbols():
    df = load_file(io.BytesIO(b"Order Date,Net $ (INR)\n01/02/2024,100\n02/02/2024,200\n"), "a.csv")
    full_pipeline(df)


# ---------------------------------------------------------------- review fixes
def test_names_that_collide_after_stripping_are_made_unique():
    df = load_file(io.BytesIO(b"a, a ,b\n1,2,3\n4,5,6\n"), "a.csv")
    assert list(df.columns) == ["a", "a.1", "b"]


def test_blank_column_name_gets_a_label():
    df = load_file(io.BytesIO(b"a,,c\n1,2,3\n4,5,6\n"), "a.csv")
    assert df.columns.is_unique and all(df.columns)


def test_mean_or_median_on_text_column_gives_clear_error():
    from core.clean import apply_step
    df = pd.DataFrame({"x": ["a", "b", None, "c"]})
    for step in ({"op": "impute", "col": "x", "method": "mean"},
                 {"op": "impute", "col": "x", "method": "median"},
                 {"op": "cap_outliers", "col": "x"}):
        with pytest.raises(ValueError, match="not a numeric column"):
            apply_step(df, step)


def test_impute_keeps_integer_dtype_when_nothing_is_missing():
    from core.clean import apply_step
    df = pd.DataFrame({"n": [1, 2, 3, 4]})
    out = apply_step(df, {"op": "impute", "col": "n", "method": "median"})
    assert pd.api.types.is_integer_dtype(out["n"])


def test_date_order_is_detected():
    from core.utils import guess_dayfirst
    assert guess_dayfirst(pd.Series(["31/01/2024", "02/03/2024"])) is True
    assert guess_dayfirst(pd.Series(["01/31/2024", "02/03/2024"])) is False
    assert guess_dayfirst(pd.Series(["02/03/2024", "04/05/2024"])) is True   # ambiguous: day-first
    df = pd.DataFrame({"d": ["01/31/2024", "02/15/2024", "03/20/2024"] * 4})
    step = [s for s in suggest_fixes(profile(df)) if s["op"] == "convert_datetime"][0]
    assert step["dayfirst"] is False
    assert replay(df, [step])["d"].iloc[0].month == 1
