"""Outlier detection: per-column rules (IQR, z-score) and a multi-column Isolation Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import eda
from core.clean import outlier_mask


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per numeric column: how many outliers each rule finds and the IQR limits."""
    rows = []
    for c in eda.numeric_cols(df):
        x = df[c].astype(float)
        if x.dropna().nunique() < 3:
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        rows.append({"column": c, "iqr_outliers": int(outlier_mask(x, "iqr").sum()),
                     "zscore_outliers": int(outlier_mask(x, "zscore").sum()),
                     "lower_limit": round(q1 - 1.5 * iqr, 3), "upper_limit": round(q3 + 1.5 * iqr, 3),
                     "min": round(x.min(), 3), "max": round(x.max(), 3)})
    return pd.DataFrame(rows)


def isolation_outliers(df: pd.DataFrame, contamination: float = 0.02, max_cols: int = 12) -> pd.DataFrame:
    """Rows that look unusual when ALL numeric columns are considered together.

    Returns the flagged rows (original index kept) with an `anomaly_score` column, most unusual first.
    """
    from sklearn.ensemble import IsolationForest
    cols = eda.numeric_cols(df)[:max_cols]
    if len(cols) < 1 or len(df) < 20:
        return pd.DataFrame()
    data = df[cols].astype(float)
    data = data.fillna(data.median())
    data = data.loc[:, data.std() > 0]
    if data.shape[1] == 0:
        return pd.DataFrame()
    model = IsolationForest(contamination=min(max(contamination, 0.001), 0.25), random_state=0, n_estimators=150)
    flag = model.fit_predict(data)
    score = -model.score_samples(data)               # higher = more unusual
    out = df.loc[flag == -1].copy()
    out.insert(0, "anomaly_score", np.round(score[flag == -1], 3))
    return out.sort_values("anomaly_score", ascending=False)
