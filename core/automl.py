"""One-click baseline models with guardrails that explain when NOT to trust the result."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import eda
from core.utils import is_id_like

MAX_ROWS = 20_000


@dataclass
class AutoMLResult:
    task: str
    target: str
    metric_name: str
    leaderboard: pd.DataFrame
    best: str
    importances: pd.DataFrame
    predictions: pd.DataFrame
    features: list[str]
    warnings: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0


def detect_task(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return "classification"
    if s.nunique(dropna=True) <= 8 and (s.dropna() % 1 == 0).all():
        return "classification"
    return "regression"


def candidate_targets(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if is_id_like(df[c], c) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        n = df[c].nunique(dropna=True)
        if n < 2 or df[c].isna().mean() > 0.4:
            continue
        if detect_task(df[c]) == "classification" and n > 20:
            continue
        out.append(c)
    return out


def _prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    notes, X = [], pd.DataFrame(index=df.index)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if is_id_like(s, c):
            notes.append(f"Left out '{c}' (looks like an identifier).")
        elif s.isna().mean() > 0.6:
            notes.append(f"Left out '{c}' (over 60% missing).")
        elif s.nunique(dropna=True) < 2:
            continue
        elif pd.api.types.is_datetime64_any_dtype(s):
            X[f"{c} (year)"], X[f"{c} (month)"] = s.dt.year, s.dt.month
            X[f"{c} (weekday)"] = s.dt.dayofweek
        elif pd.api.types.is_bool_dtype(s):
            X[c] = s.map(lambda v: np.nan if pd.isna(v) else float(v))
        elif pd.api.types.is_numeric_dtype(s):
            X[c] = s.astype(float)
        elif s.nunique(dropna=True) > 50:
            notes.append(f"Left out '{c}' (too many different values).")
        else:
            X[c] = s.astype("object").where(s.notna(), np.nan)
    return X, notes


def _leakage(X: pd.DataFrame, y: pd.Series, task: str) -> list[str]:
    """Features that give the answer away (so the model would look great but be useless in practice)."""
    warn = []
    yn = y.astype(float) if task == "regression" else pd.Series(pd.factorize(y)[0], index=y.index).where(y.notna())
    for c in X.columns:
        s = X[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                with np.errstate(all="ignore"):
                    r = s.corr(yn)
                if pd.notna(r) and abs(r) > 0.97:
                    warn.append(f"'{c}' is almost identical to the target (correlation {r:.2f}). "
                                "It may contain the answer; remove it if it would not be known in real use.")
            elif task == "classification" and s.nunique() > 1:
                tab = pd.crosstab(s, y)
                purity = tab.max(axis=1).sum() / tab.values.sum()
                if purity > 0.99 and s.nunique() < len(s) * 0.5:
                    warn.append(f"'{c}' predicts the target almost perfectly by itself. Check it is not a leak.")
        except Exception:
            continue
    return warn


def run(df: pd.DataFrame, target: str, seed: int = 0) -> AutoMLResult:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")
    data = df.dropna(subset=[target]).copy()
    if len(data) < 30:
        raise ValueError("Need at least 30 rows with a known target to train a model.")
    warnings: list[str] = []
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=seed)
        warnings.append(f"Used a random sample of {MAX_ROWS:,} rows to keep this fast.")

    task = detect_task(data[target])
    y = data[target]
    X, notes = _prepare_features(data, target)
    warnings += notes
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns left after removing identifiers and empty columns.")
    if len(data) < 100:
        warnings.append(f"Only {len(data)} rows: results will vary a lot. Treat them as a rough guide.")
    warnings += _leakage(X, y, task)

    if task == "classification":
        y = y.astype(str)
        counts = y.value_counts()
        if counts.min() < 5:
            rare = counts[counts < 5].index.tolist()
            data_mask = ~y.isin(rare)
            warnings.append(f"Removed classes with fewer than 5 rows: {', '.join(rare[:5])}.")
            X, y = X[data_mask], y[data_mask]
            counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError("The target needs at least two classes with 5+ rows each.")
        if counts.iloc[-1] / len(y) < 0.1:
            warnings.append(f"Imbalanced target: '{counts.index[-1]}' is only {counts.iloc[-1] / len(y):.0%} of rows. "
                            "Accuracy alone can be misleading; check the F1 score.")

    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]

    def pre(scale: bool):
        n_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
        return ColumnTransformer([
            ("num", Pipeline(n_steps), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])

    if task == "classification":
        models = {"Baseline (always the most common class)": (DummyClassifier(strategy="most_frequent"), True),
                  "Logistic regression": (LogisticRegression(max_iter=1000), True),
                  "Random forest": (RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1), False),
                  "Gradient boosting": (HistGradientBoostingClassifier(random_state=seed), False)}
        scoring, metric = "f1_weighted", "F1 (weighted)"
    else:
        models = {"Baseline (always the average)": (DummyRegressor(strategy="mean"), True),
                  "Ridge regression": (Ridge(), True),
                  "Random forest": (RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1), False),
                  "Gradient boosting": (HistGradientBoostingRegressor(random_state=seed), False)}
        scoring, metric = "r2", "R²"

    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=strat)
    cv = max(2, min(5, len(Xtr) // 20))

    rows, fitted = [], {}
    for name, (est, scale) in models.items():
        pipe = Pipeline([("pre", pre(scale)), ("model", est)])
        try:
            cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring)
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
        except Exception as e:                      # one failing model must not kill the run
            warnings.append(f"{name} could not be trained: {e}")
            continue
        fitted[name] = (pipe, pred)
        if task == "classification":
            rows.append({"model": name, f"cv {metric}": round(cvs.mean(), 3), "cv spread": round(cvs.std(), 3),
                         "test accuracy": round(accuracy_score(yte, pred), 3),
                         "test F1 (weighted)": round(f1_score(yte, pred, average="weighted"), 3)})
        else:
            rows.append({"model": name, f"cv {metric}": round(cvs.mean(), 3), "cv spread": round(cvs.std(), 3),
                         "test R²": round(r2_score(yte, pred), 3),
                         "test MAE": round(mean_absolute_error(yte, pred), 3)})
    board = pd.DataFrame(rows)
    if board.shape[0] < 2:
        raise ValueError("Models could not be trained on this data.")
    key = f"cv {metric}"
    board = board.sort_values(key, ascending=False).reset_index(drop=True)
    real = board[~board["model"].str.startswith("Baseline")]
    best = real.iloc[0]["model"]
    base_score = float(board[board["model"].str.startswith("Baseline")][key].iloc[0])
    best_score = float(real.iloc[0][key])

    pipe, pred = fitted[best]
    imp = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=seed, scoring=scoring)
    importances = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
                   .sort_values("importance", ascending=False).reset_index(drop=True))
    importances["importance"] = importances["importance"].round(4)

    summary = []
    if task == "classification":
        summary.append(f"Predicting **{target}** (categories). Best model: **{best}**, "
                       f"correct on {board.loc[board.model == best, 'test accuracy'].iloc[0]:.0%} of unseen rows.")
    else:
        mae = board.loc[board.model == best, "test MAE"].iloc[0]
        r2 = board.loc[board.model == best, "test R²"].iloc[0]
        summary.append(f"Predicting **{target}** (a number). Best model: **{best}**. It explains "
                       f"{max(r2, 0):.0%} of the variation and is off by {mae:,.2f} on average.")
    worse_acc = False
    if task == "classification":
        acc = board.set_index("model")["test accuracy"]
        worse_acc = acc[best] <= acc[board[board.model.str.startswith("Baseline")].model.iloc[0]]
    if best_score - base_score < 0.05 or worse_acc:
        warnings.append("The best model barely beats the baseline guess. These columns may not predict "
                        f"'{target}' well, so do not rely on this model.")
    top = importances[importances.importance > 0].head(3)["feature"].tolist()
    if top:
        summary.append("Most influential columns: " + ", ".join(f"**{t}**" for t in top) + ".")
    if board.loc[board.model == best, "cv spread"].iloc[0] > 0.1:
        warnings.append("Scores differ a lot between validation folds, so the result is unstable.")
    summary.append("Importance shows which columns the model used, not what causes the target.")

    preds = pd.DataFrame({"actual": yte.values, "predicted": pred}, index=yte.index)
    if task == "regression":
        preds["error"] = (preds.predicted - preds.actual).round(3)
    else:
        preds["correct"] = preds.actual == preds.predicted
    return AutoMLResult(task, target, metric, board, best, importances, preds, list(X.columns),
                        warnings, summary, len(Xtr), len(Xte))
