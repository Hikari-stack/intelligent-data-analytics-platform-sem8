import io
from pathlib import Path

import pandas as pd
import streamlit as st

from app import state, ui
from core.ingest import IngestError, list_sheets, load_file
from db.repo import list_datasets
from db.session import SessionLocal

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "messy_sales.csv"


def render():
    s = st.session_state
    if s.raw_df is None:
        ui.intro("From messy file to clear answers",
                 "Upload a spreadsheet, see exactly what is wrong with it, fix it with one click, and ask "
                 "questions in plain English. Every step is explained and can be replayed on the next file.")
        c = st.columns(4)
        with c[0]: ui.card("Quality report", "A 0–100 score and a plain-language list of what is wrong.")
        with c[1]: ui.card("Guided cleaning", "One-click fixes with reasons, undo/redo and a replayable log.")
        with c[2]: ui.card("Ask your data", "Type a question. Get an answer, a chart and the code behind it.")
        with c[3]: ui.card("Predict", "Train baseline models and get warned when not to trust them.")
        st.write("")
    else:
        ui.page_header("Upload", "Load a different file at any time. Your cleaning steps reset for the new dataset.")

    st.toggle("Clean the data automatically after loading", value=True, key="auto_clean",
              help="Applies the safe suggested fixes straight away. Every step is listed on the Clean page and can be undone.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        up = st.file_uploader("Drop a CSV, TSV, TXT or Excel file (max 100 MB)",
                              type=["csv", "tsv", "txt", "xlsx", "xls"])
        if up is not None:
            data = up.getvalue()
            sheet, readable = None, True
            if up.name.lower().endswith((".xlsx", ".xls")):
                try:
                    sheets = list_sheets(data)
                    if len(sheets) > 1:
                        sheet = st.selectbox("This workbook has several sheets. Pick one:", sheets)
                except IngestError as e:
                    st.error(str(e))
                    readable = False
            key = f"{up.name}|{sheet}"
            if readable and s.last_upload_key != key:
                try:
                    label = up.name if sheet is None else f"{up.name} [{sheet}]"
                    loaded = load_file(data, up.name, sheet=sheet)
                    state.set_dataset(loaded, label)
                    s.load_notice = _load_notice(loaded.attrs)
                    s.clean_notice = _auto_clean_notice() if s.auto_clean else None
                    s.last_upload_key = key
                    st.rerun()  # so the sidebar and stepper pick up the new dataset straight away
                except IngestError as e:
                    st.error(str(e))
    with right:
        st.markdown("**No file handy?**")
        if st.button("Use the sample messy dataset", width="stretch"):
            state.set_dataset(load_file(io.BytesIO(SAMPLE_PATH.read_bytes()), SAMPLE_PATH.name), SAMPLE_PATH.name)
            s.last_upload_key = None
            s.load_notice = None
            s.clean_notice = _auto_clean_notice() if s.auto_clean else None
            st.rerun()
        st.caption("400 sales rows with typical real-world problems: numbers stored as text, mixed date formats, "
                   "inconsistent spellings, duplicates and outliers.")
        _recent()

    if s.raw_df is None:
        return
    df = state.current_df()
    if s.get("load_notice"):
        st.info(s.load_notice)
    if s.get("clean_notice"):
        st.success(s.clean_notice)
    elif not s.steps:
        st.button("Clean the data automatically", on_click=_run_auto_clean, type="primary")
    st.divider()
    st.subheader(f"Preview · {s.filename}" + (" (cleaned)" if s.steps else ""))
    c = st.columns(4)
    with c[0]: ui.kpi("Rows", f"{len(df):,}")
    with c[1]: ui.kpi("Columns", df.shape[1])
    with c[2]: ui.kpi("Missing cells", f"{int(df.isna().sum().sum()):,}")
    with c[3]: ui.kpi("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    st.write("")
    q = st.text_input("Search the preview", placeholder="Type to filter rows (searches every column)")
    view = df
    if q:
        mask = df.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
        view = df[mask]
        st.caption(f"{len(view):,} matching rows (showing up to 200)")
    st.dataframe(view.head(200), width="stretch", height=330)
    with st.expander("Column types as loaded"):
        st.caption("Text columns may secretly be numbers or dates; the Quality Report finds these.")
        st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "non-empty": df.notna().sum()}),
                     width="stretch")
    st.success("Next: open **Quality report** in the sidebar to see what needs fixing.")


def _auto_clean_notice() -> str | None:
    r = state.auto_clean()
    if not r["steps"]:
        return "The data already looked clean, so no changes were needed."
    msg = (f"Cleaned automatically: {r['steps']} step{'s' if r['steps'] != 1 else ''} applied, "
           f"quality score {r['before']:g} to {r['after']:g}. Every step is listed on the Clean page and can be undone.")
    if r["left_alone"]:
        msg += " Left unfilled because filling them would invent data: " + ", ".join(r["left_alone"]) + "."
    if r["outlier_cols"]:
        msg += " Unusual values were not changed; review them on the Clean page (Outliers tab)."
    return msg


def _run_auto_clean():
    st.session_state.clean_notice = _auto_clean_notice()


def _load_notice(attrs) -> str | None:
    """Plain-language note about rows that were repaired or skipped while reading the file."""
    parts = []
    rep = attrs.get("repaired_lines")
    if rep:
        shown = ", ".join(f"line {ln} (merged into '{col}')" for ln, col in rep[:6]) + (" ..." if len(rep) > 6 else "")
        parts.append(f"{len(rep)} row{'s' if len(rep) != 1 else ''} contained extra commas and "
                     f"{'were' if len(rep) != 1 else 'was'} repaired automatically: {shown}. "
                     "You can check them in the data table below.")
    lines = attrs.get("skipped_lines")
    if lines:
        shown = ", ".join(str(n) for n in lines[:8]) + (" ..." if len(lines) > 8 else "")
        parts.append(f"{len(lines)} row{'s' if len(lines) != 1 else ''} had more fields than the header and could not "
                     f"be repaired, so {'they were' if len(lines) != 1 else 'it was'} skipped (file lines: {shown}).")
    return " ".join(parts) or None


def _recent():
    try:
        state.db_ready()
        with SessionLocal() as db:
            rows = list_datasets(db)[:3]
        if rows:
            st.markdown("**Recent datasets**")
            for d in rows:
                st.caption(f"{d.name} · {d.n_rows:,} rows · {d.created_at:%d %b %H:%M}")
            st.caption("Open the History page to load one.")
    except Exception:
        pass
