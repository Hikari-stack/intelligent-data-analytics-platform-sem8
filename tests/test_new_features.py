import numpy as np
import pandas as pd
import pytest

from core import automl, nlq, outliers, report, timeseries
from core.clean import apply_step, replay
from core.profile import describe_step, profile


@pytest.fixture()
def sales():
    rng = np.random.default_rng(0)
    n = 240
    price = rng.normal(100, 20, n)
    df = pd.DataFrame({
        "OrderID": range(1, n + 1),
        "OrderDate": pd.date_range("2023-01-01", periods=n, freq="3D"),
        "Region": rng.choice(["East", "West", "North"], n),
        "Units": rng.integers(1, 10, n).astype(float),
        "Price": price,
    })
    df["Revenue"] = df["Units"] * df["Price"] + rng.normal(0, 5, n)
    return df


# ------------------------------------------------------------------ cleaning ops
def test_filter_rows_numeric_and_text(sales):
    out = apply_step(sales, {"op": "filter_rows", "col": "Units", "operator": ">=", "value": "5"})
    assert (out.Units >= 5).all() and len(out) < len(sales)
    out = apply_step(sales, {"op": "filter_rows", "col": "Region", "operator": "==", "value": "East"})
    assert set(out.Region) == {"East"}
    out = apply_step(sales, {"op": "filter_rows", "col": "Region", "operator": "contains", "value": "ES"})
    assert set(out.Region) == {"West"}


def test_filter_rows_errors_are_friendly(sales):
    with pytest.raises(ValueError, match="must be a number"):
        apply_step(sales, {"op": "filter_rows", "col": "Units", "operator": ">", "value": "abc"})
    with pytest.raises(ValueError, match="every row"):
        apply_step(sales, {"op": "filter_rows", "col": "Units", "operator": ">", "value": "999"})
    with pytest.raises(ValueError, match="Unknown filter"):
        apply_step(sales, {"op": "filter_rows", "col": "Units", "operator": "~", "value": "1"})


def test_find_replace(sales):
    out = apply_step(sales, {"op": "find_replace", "col": "Region", "find": "East", "replace": "E"})
    assert "E" in set(out.Region) and "East" not in set(out.Region)
    with pytest.raises(ValueError):
        apply_step(sales, {"op": "find_replace", "col": "Region", "find": ""})


def test_drop_outlier_rows(sales):
    d = sales.copy()
    d.loc[0, "Price"] = 10_000
    out = apply_step(d, {"op": "drop_outlier_rows", "col": "Price", "method": "iqr"})
    assert len(out) < len(d) and out.Price.max() < 1000
    with pytest.raises(ValueError, match="not a numeric"):
        apply_step(d, {"op": "drop_outlier_rows", "col": "Region"})


def test_knn_impute_uses_similar_rows(sales):
    d = sales.copy()
    truth = d.loc[[3, 10, 50], "Revenue"].copy()
    d.loc[[3, 10, 50], "Revenue"] = np.nan
    out = apply_step(d, {"op": "impute", "col": "Revenue", "method": "knn", "k": 5})
    assert out.Revenue.notna().all()
    knn_err = (out.loc[truth.index, "Revenue"] - truth).abs().mean()
    mean_err = (d.Revenue.mean() - truth).abs().mean()
    assert knn_err < mean_err          # neighbours beat a global average when columns are related
    assert "KNN" in describe_step({"op": "impute", "col": "Revenue", "method": "knn", "k": 5})


def test_new_ops_replay_and_json_roundtrip(sales):
    import json
    steps = [{"op": "filter_rows", "col": "Units", "operator": ">", "value": "2"},
             {"op": "drop_outlier_rows", "col": "Price"}]
    steps = json.loads(json.dumps(steps))
    assert len(replay(sales, steps)) <= len(sales)
    for s in steps:
        assert describe_step(s)


# ------------------------------------------------------------------ nlq
def test_nlq_groupby_and_top(sales):
    a = nlq.ask(sales, "average revenue by region")
    assert a.ok and a.table is not None and set(a.table["Region"]) == {"East", "West", "North"}
    top = nlq.ask(sales, "which region has the highest revenue")
    expect = sales.groupby("Region")["Revenue"].sum().idxmax()
    assert top.ok and expect in top.text
    low = nlq.ask(sales, "which region has the lowest revenue")
    assert sales.groupby("Region")["Revenue"].sum().idxmin() in low.text


def test_nlq_filter_count_corr_time(sales):
    n = (sales.Region == "East").sum()
    assert f"**{n}**" in nlq.ask(sales, "how many rows where region is east").text
    total = sales.loc[sales.Region == "West", "Revenue"].sum()
    assert f"{total:,.2f}" in nlq.ask(sales, "total revenue in the west").text
    c = nlq.ask(sales, "correlation between units and revenue")
    assert c.ok and "positively" in c.text
    t = nlq.ask(sales, "monthly revenue")
    assert t.ok and t.figure is not None


def test_nlq_unknown_and_empty(sales):
    for q in ["", "blah blah", "tell me a joke"]:
        a = nlq.ask(sales, q)
        assert not a.ok and a.suggestions is not None


def test_nlq_camelcase_columns():
    df = pd.DataFrame({"CustomerAge": [20, 30, 40, 50] * 10, "Segment": ["a", "b"] * 20})
    assert nlq.ask(df, "average customer age by segment").ok


# ------------------------------------------------------------------ outliers / time series
def test_outlier_summary_and_isolation(sales):
    d = sales.copy()
    d.loc[5, ["Price", "Revenue"]] = [5000, 90000]
    summ = outliers.column_summary(d)
    assert summ.set_index("column").loc["Price", "iqr_outliers"] >= 1
    flagged = outliers.isolation_outliers(d, 0.02)
    assert 5 in flagged.index and flagged.index[0] == 5
    assert outliers.isolation_outliers(d.head(5)).empty


def test_timeseries(sales):
    s = timeseries.aggregate(sales, "OrderDate", "Revenue", "Month", "sum")
    assert len(s) >= 12 and s.index.is_monotonic_increasing
    res = timeseries.analyse(s, 3)
    assert res["summary"] and timeseries.figure(res) is not None
    with pytest.raises(ValueError):
        timeseries.aggregate(sales, "OrderDate", "Revenue", "Fortnight")


# ------------------------------------------------------------------ automl
def test_automl_regression_beats_baseline(sales):
    r = automl.run(sales, "Revenue")
    assert r.task == "regression" and "OrderID" not in r.features
    base = r.leaderboard[r.leaderboard.model.str.startswith("Baseline")]["test R²"].iloc[0]
    best = r.leaderboard.set_index("model").loc[r.best, "test R²"]
    assert best > base + 0.5
    assert r.importances.iloc[0]["feature"] in ("Units", "Price")


def test_automl_warns_about_leakage_and_noise(sales):
    d = sales.copy()
    d["RevenueCopy"] = d["Revenue"] * 2
    assert any("RevenueCopy" in w for w in automl.run(d, "Revenue").warnings)
    d = sales.copy()
    d["Noise"] = np.random.default_rng(1).normal(size=len(d))
    r = automl.run(d[["Region", "Noise"]].assign(Target=np.random.default_rng(2).choice(["x", "y"], len(d))), "Target")
    assert any("baseline" in w.lower() for w in r.warnings)


def test_automl_guards():
    with pytest.raises(ValueError, match="at least 30"):
        automl.run(pd.DataFrame({"a": range(10), "b": range(10)}), "b")
    with pytest.raises(ValueError):
        automl.run(pd.DataFrame({"a": range(50)}), "zzz")
    assert automl.detect_task(pd.Series([0, 1, 0, 1] * 5)) == "classification"
    assert automl.detect_task(pd.Series(np.linspace(0, 1, 50))) == "regression"


# ------------------------------------------------------------------ report
def test_report_html(sales):
    p = profile(sales)
    html = report.build_html(sales, sales, [{"op": "drop_duplicates"}], p, p, "<b>x</b>.csv", None, ["Model **ok**"])
    assert "<!doctype html>" in html and "Remove duplicate rows" in html
    assert "&lt;b&gt;x&lt;/b&gt;.csv" in html          # file name is escaped
    assert "Model ok" in html
