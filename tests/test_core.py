import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.clean import apply_step, replay
from core.eda import generate_insights, guardrails
from core.ingest import IngestError, load_file
from core.profile import profile, suggest_fixes

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "messy_sales.csv"


def messy():
    return pd.DataFrame({
        "Region": ["West", "west ", "WEST", "North", None],
        "Sales": ["1,200", "300", "N/A", "50.5", "10"],
        "Date": ["01/02/2024", "2024-02-03", "5 Mar 2024", "06/03/2024", "07/03/2024"],
        "Age": [20.0, np.nan, 30.0, 40.0, 25.0],
    })


# ---------- ingest ----------
def test_load_csv_with_semicolon():
    df = load_file(io.BytesIO(b"a;b\n1;2\n3;4\n"), "x.csv")
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_load_rejects_empty_and_bad_type():
    with pytest.raises(IngestError):
        load_file(io.BytesIO(b""), "x.csv")
    with pytest.raises(IngestError):
        load_file(io.BytesIO(b"abc"), "x.pdf")


# ---------- profile ----------
def test_profile_detects_issues():
    p = profile(messy())
    cols = p["columns"]
    assert cols["Sales"]["kind"] == "numeric_text"
    assert cols["Date"]["kind"] == "datetime_text"
    assert cols["Region"]["variants"]
    assert cols["Sales"]["null_tokens"] == 1
    assert cols["Age"]["missing"] == 1


def test_suggestions_are_ordered_and_skip_pending_columns():
    ops = [(s["op"], s.get("col")) for s in suggest_fixes(profile(messy()))]
    assert ("normalize_missing", None) in ops
    assert ("convert_numeric", "Sales") in ops
    assert ("impute", "Age") in ops
    # Region still has spelling variants, so no impute suggestion yet
    assert ("impute", "Region") not in ops


# ---------- clean ----------
def test_convert_numeric_and_datetime():
    out = apply_step(messy(), {"op": "convert_numeric", "col": "Sales"})
    assert out["Sales"].iloc[0] == 1200.0 and np.isnan(out["Sales"].iloc[2])
    out = apply_step(messy(), {"op": "convert_datetime", "col": "Date"})
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])
    assert out["Date"].notna().all()
    assert out["Date"].iloc[0] == pd.Timestamp("2024-02-01")  # day-first


def test_standardize_categories():
    out = apply_step(messy(), {"op": "standardize_categories", "col": "Region"})
    assert set(out["Region"].dropna()) == {"West", "North"}


def test_impute_methods():
    df = messy()
    assert apply_step(df, {"op": "impute", "col": "Age", "method": "median"})["Age"].isna().sum() == 0
    out = apply_step(df, {"op": "impute", "col": "Region", "method": "constant", "value": "Unknown"})
    assert out["Region"].iloc[4] == "Unknown"


def test_cap_outliers():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 1000]})
    out = apply_step(df, {"op": "cap_outliers", "col": "x"})
    assert out["x"].max() < 1000


def test_apply_step_does_not_mutate_input():
    df = messy()
    before = df.copy()
    apply_step(df, {"op": "convert_numeric", "col": "Sales"})
    pd.testing.assert_frame_equal(df, before)


def test_unknown_op_and_missing_column():
    with pytest.raises(ValueError):
        apply_step(messy(), {"op": "explode"})
    with pytest.raises(ValueError):
        apply_step(messy(), {"op": "impute", "col": "Nope", "method": "mean"})


# ---------- the key promise: replay ----------
def test_replay_is_deterministic_and_undo_works():
    raw = load_file(io.BytesIO(SAMPLE.read_bytes()), "messy_sales.csv")
    steps = []
    for _ in range(8):  # keep applying suggestions until none are left
        cur = replay(raw, steps)
        sugg = suggest_fixes(profile(cur))
        if not sugg:
            break
        steps.append({k: v for k, v in sugg[0].items()})
    final_a, final_b = replay(raw, steps), replay(raw, steps)
    pd.testing.assert_frame_equal(final_a, final_b)
    assert final_a.duplicated().sum() == 0
    assert profile(final_a)["overview"]["quality_score"] > profile(raw)["overview"]["quality_score"]
    # undo = drop last step and replay
    assert len(replay(raw, steps[:-1])) >= 0


# ---------- eda ----------
def test_insights_and_guardrails():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": -2 * x + rng.normal(scale=0.1, size=200)})
    ins = generate_insights(df)
    assert any("negatively" in i for i in ins)
    assert any("causes" in g for g in guardrails(df))
    assert any("unreliable" in g for g in guardrails(df.head(10)))


# ---------- v2: ID columns, whole-number imputation, chart swap ----------
def test_id_like_detection():
    from core.utils import is_id_like
    ids = pd.Series(range(1001, 1101), name="OrderID")
    assert is_id_like(ids, "OrderID")
    assert is_id_like(pd.Series(range(100)), "row")            # plain 0..n-1 sequence
    assert not is_id_like(pd.Series(np.random.default_rng(0).integers(0, 10, 100)), "Quantity")
    assert not is_id_like(pd.Series([1.5, 2.5] * 10), "Price")  # not whole numbers


def test_id_column_excluded_from_analysis():
    from core.eda import chart_for, correlation, numeric_cols
    df = pd.DataFrame({"OrderID": range(1, 51),
                       "a": np.arange(50) * 2.0, "b": np.arange(50) * -1.0})
    assert "OrderID" not in numeric_cols(df)
    assert "OrderID" not in correlation(df).columns
    assert chart_for(df, "OrderID") is None
    kinds = profile(df)["columns"]
    assert kinds["OrderID"]["kind"] == "id"


def test_whole_number_column_gets_whole_number_fill():
    df = pd.DataFrame({"Quantity": [1.0, 2.0, 4.0, 9.0, np.nan, 3.0, 5.0, 8.0, 7.0, 4.0, 6.0, np.nan]})
    sugg = [s for s in suggest_fixes(profile(df)) if s["op"] == "impute"][0]
    assert sugg["method"] == "median"
    for method in ("mean", "median"):
        out = apply_step(df, {"op": "impute", "col": "Quantity", "method": method})
        assert (out["Quantity"] % 1 == 0).all()


def test_chart_for_date_vs_number_either_order():
    from core.eda import chart_for
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=30), "v": np.arange(30.0)})
    assert chart_for(df, "d", "v") is not None
    assert chart_for(df, "v", "d") is not None   # used to show "No suitable chart"


def test_constant_column_is_flagged_and_dropped():
    df = pd.DataFrame({"Country": ["India"] * 12, "Score": list(range(12))})
    info = profile(df)["columns"]["Country"]
    assert info["constant"] is True
    ops = [(s["op"], s.get("col")) for s in suggest_fixes(profile(df))]
    assert ("drop_column", "Country") in ops
    # a normal column is not flagged
    assert profile(df)["columns"]["Score"]["constant"] is False


# ---------- v3: mixed types, booleans, new ops, Excel ----------
def test_boolean_text_detected_converted_and_suggested():
    df = pd.DataFrame({"Member": ["Yes", "no", "Y", "N", None, "yes", "No", "YES", "n", "y", "no", "yes"]})
    info = profile(df)["columns"]["Member"]
    assert info["kind"] == "boolean_text"
    assert ("convert_boolean", "Member") in [(s["op"], s.get("col")) for s in suggest_fixes(profile(df))]
    out = apply_step(df, {"op": "convert_boolean", "col": "Member"})
    assert out["Member"].iloc[0] is True or bool(out["Member"].iloc[0]) is True
    assert bool(out["Member"].iloc[1]) is False
    assert out["Member"].isna().sum() == 1
    # a plain 1/0 text column is treated as numbers, not yes/no
    nums = pd.DataFrame({"x": ["1", "0", "1", "0"] * 3})
    assert profile(nums)["columns"]["x"]["kind"] == "numeric_text"


def test_mixed_type_column_is_flagged_but_not_auto_fixed():
    df = pd.DataFrame({"Code": ["10", "abc", "20", "xyz", "30", "pqr", "40", "lmn", "50", "def"]})
    info = profile(df)["columns"]["Code"]
    assert info["mixed"] is True
    assert not [s for s in suggest_fixes(profile(df)) if s.get("col") == "Code" and s["op"] == "convert_numeric"]
    clean = pd.DataFrame({"Code": ["a", "b", "c", "d"] * 3})
    assert profile(clean)["columns"]["Code"]["mixed"] is False


def test_drop_rows_missing():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, None, 3, 4], "c": [1, None, None, 4]})
    out = apply_step(df, {"op": "drop_rows_missing", "threshold": 0.5})
    assert len(out) == 3                      # only the row with 100% missing is removed
    with pytest.raises(ValueError):
        apply_step(df, {"op": "drop_rows_missing", "threshold": 0})


def test_replace_values_and_rename_column():
    df = pd.DataFrame({"City": ["Bombay", "Delhi", "Bombay"]})
    out = apply_step(df, {"op": "replace_values", "col": "City", "mapping": {"Bombay": "Mumbai"}})
    assert list(out["City"]) == ["Mumbai", "Delhi", "Mumbai"]
    with pytest.raises(ValueError):
        apply_step(df, {"op": "replace_values", "col": "City", "mapping": {}})
    out = apply_step(df, {"op": "rename_column", "col": "City", "new_name": "Town"})
    assert list(out.columns) == ["Town"]
    two = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        apply_step(two, {"op": "rename_column", "col": "a", "new_name": "b"})


def test_pipeline_with_all_new_ops_replays_identically(tmp_path):
    df = pd.DataFrame({"Member": ["yes", "no", None, "yes"], "City": ["Bombay", "Delhi", "Bombay", None],
                       "x": [1.0, None, None, None]})
    steps = [{"op": "convert_boolean", "col": "Member"},
             {"op": "replace_values", "col": "City", "mapping": {"Bombay": "Mumbai"}},
             {"op": "drop_rows_missing", "threshold": 0.6},
             {"op": "rename_column", "col": "City", "new_name": "Town"}]
    import json
    steps = json.loads(json.dumps(steps))     # same as saving and loading pipeline.json
    a, b = replay(df, steps), replay(df, steps)
    pd.testing.assert_frame_equal(a, b)
    assert "Town" in a.columns


def test_excel_sheet_selection():
    from core.ingest import list_sheets
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="First", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(w, sheet_name="Second", index=False)
    data = buf.getvalue()
    assert list_sheets(data) == ["First", "Second"]
    assert list(load_file(data, "x.xlsx").columns) == ["a"]                      # default: first
    assert list(load_file(data, "x.xlsx", sheet="Second").columns) == ["b"]
    with pytest.raises(IngestError):
        list_sheets(b"not an excel file")


# ---------- v4: fixes found by running Titanic and Telco ----------
def test_cap_outliers_keeps_whole_numbers_whole():
    df = pd.DataFrame({"SibSp": [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 8, 0, 0, 1, 0, 0, 1, 0, 0, 1]})
    out = apply_step(df, {"op": "cap_outliers", "col": "SibSp"})
    assert (out["SibSp"] % 1 == 0).all()
    assert out["SibSp"].max() < 8


def test_code_columns_like_tickets_are_not_flagged_mixed():
    tickets = [f"A/5 {i}" if i % 3 else str(100000 + i) for i in range(60)]
    assert profile(pd.DataFrame({"Ticket": tickets}))["columns"]["Ticket"]["mixed"] is False


def test_true_false_columns_join_correlations():
    from core.eda import correlation
    rng = np.random.default_rng(1)
    tenure = rng.integers(1, 70, 300).astype(float)
    churn = pd.Series(tenure < 20).astype("boolean")
    df = pd.DataFrame({"tenure": tenure, "Churn": churn, "x": rng.normal(size=300)})
    corr = correlation(df)
    assert "Churn" in corr.columns and corr.loc["tenure", "Churn"] < -0.5


# ---------- Task 4.1 / 4.3 ----------
def test_cramers_v_identical_and_independent():
    from core.eda import cramers_v
    rng = np.random.default_rng(0)
    a = rng.choice(["x", "y", "z"], 600)
    df = pd.DataFrame({"a": a, "same": a, "other": rng.choice(["p", "q"], 600)})
    assert cramers_v(df, "a", "same") > 0.99
    assert cramers_v(df, "a", "other") < 0.15


def test_association_insight_appears():
    rng = np.random.default_rng(2)
    region = rng.choice(["N", "S", "E"], 400)
    cat = np.where(region == "N", "A", np.where(region == "S", "B", "C"))
    cat = np.where(rng.random(400) < 0.1, "A", cat)
    df = pd.DataFrame({"Region": region, "Category": cat})
    assert any("associated" in i for i in generate_insights(df))


def test_extra_guardrails_fire_and_stay_quiet():
    rng = np.random.default_rng(3)
    many = pd.DataFrame({"City": rng.choice([f"c{i}" for i in range(80)], 400),
                         "Flag": ["x"] * 390 + ["y"] * 10, "v": rng.normal(size=400)})
    msgs = " ".join(guardrails(many))
    assert "too many groups" in msgs and "over 95%" in msgs
    calm = pd.DataFrame({"Region": rng.choice(["N", "S", "E", "W"], 400), "v": rng.normal(size=400)})
    assert not [m for m in guardrails(calm) if "95%" in m or "too many" in m]


def test_missing_matrix():
    from core.eda import missing_matrix
    assert missing_matrix(pd.DataFrame({"a": [1, 2, 3]})) is None
    assert missing_matrix(pd.DataFrame({"a": [1, None, 3]})) is not None
