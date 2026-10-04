# Intelligent Data Analytics Platform

Guided, explainable, reproducible data analysis for non-experts.
**Upload → Quality report → Clean (replayable pipeline) → Explore → Insights → Export**

## Setup (once)

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/main.py
```

Click **Use the sample messy dataset** on the first page to try it immediately.

## Test

```bash
python -m pytest -q
```

## Features

| Area | What it does |
|------|--------------|
| Quality report | 0-100 score gauge, per-column issue chips, missing-% bars |
| Clean | One-click suggested fixes with reasons; manual tools (fill missing incl. KNN, filter rows, find & replace, convert type, rename, drop); outlier review (cap / remove rows, Isolation Forest view); undo / redo; replayable pipeline JSON |
| Explore | Insights + cautions, column drill-down, correlations, chart builder, searchable data table |
| Insights | **Ask your data** (plain-English questions, offline, shows the code), **Time series** (trend, moving average, seasonality), **Predict / AutoML** (baseline models, permutation importance, leakage / imbalance / "barely beats a guess" warnings) |
| Dashboard | Pin charts and answers; date and category filters; KPI tiles |
| Export & report | CSV, Excel (data + cleaning log + quality sheet), Parquet, pipeline JSON, shareable HTML report |
| History | Saved datasets and pipelines (SQLite by default, or set `DATABASE_URL` for MySQL) |

## Project layout

```
app/main.py        entry point: sidebar, stepper, page routing
app/ui.py          CSS, header, stepper, cards, gauge (no business logic)
app/state.py       session state, cached computations, callbacks
app/views/         one file per page
core/ingest.py     load CSV/Excel with friendly errors
core/profile.py    quality profiling + fix suggestions (with reasons)
core/clean.py      cleaning operations + replayable pipeline (apply_step, replay)
core/eda.py        stats, plain-language insights, guardrails, charts
core/outliers.py   per-column and multi-column outlier detection
core/timeseries.py aggregation, trend, seasonality
core/nlq.py        rule-based "ask your data" (no external service)
core/automl.py     baseline models + guardrails
core/report.py     self-contained HTML report
core/theme.py      one palette + Plotly template for every chart
db/                SQLAlchemy history (SQLite default)
tests/             pytest unit + headless UI tests
```

## How the pipeline works

Every cleaning action is a small dict, e.g. `{"op": "impute", "col": "Age", "method": "median"}`.
The cleaned data is always `replay(raw_data, steps)`, so **undo** = drop the last step,
**redo** = put it back, and a saved `pipeline.json` can be replayed on any new file with the same columns.

## Customising the look

Colours live in `.streamlit/config.toml` (Streamlit theme), `core/theme.py` (charts) and the CSS
block at the top of `app/ui.py`. Change the accent `#4F46E5` in those three places to rebrand.
