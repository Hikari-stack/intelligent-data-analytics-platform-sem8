import pandas as pd
import streamlit as st

from app import state, ui

KIND_LABEL = {"numeric": "number", "numeric_text": "number as text", "datetime": "date", "datetime_text": "date as text",
              "categorical": "category", "boolean": "yes/no", "boolean_text": "yes/no as text", "text": "text", "id": "identifier"}


def issues_for(i: dict) -> list[tuple[str, str]]:
    out = []
    if i["kind"] == "id": out.append(("identifier", "info"))
    if i["kind"] == "numeric_text": out.append(("numbers stored as text", "warn"))
    if i["kind"] == "datetime_text": out.append(("dates stored as text", "warn"))
    if i["kind"] == "boolean_text": out.append(("yes/no stored as text", "warn"))
    if i["null_tokens"]: out.append((f"{i['null_tokens']} placeholders", "warn"))
    if i["variants"]: out.append(("inconsistent spellings", "warn"))
    if i["mixed"]: out.append(("numbers and words mixed", "bad"))
    if i["constant"]: out.append(("same value everywhere", "bad"))
    if i["outlier_pct"] > 1: out.append((f"{i['outliers']} outliers", "warn"))
    return out


def render_profile(df: pd.DataFrame, title: str, skey: str, compact: bool = False):
    prof = state.cached_profile(df, st.session_state.token, skey)
    ov = prof["overview"]
    if title:
        st.subheader(title)
    g, k = st.columns([1, 2], gap="medium") if not compact else (st.container(), st.container())
    with g:
        ui.score_block(ov["quality_score"])
    with k:
        c = st.columns(2 if compact else 4)
        items = [("Rows", f"{ov['rows']:,}", ""), ("Duplicate rows", ov["duplicate_rows"], ""),
                 ("Missing cells", f"{ov['missing_cells']:,}", f"{ov['missing_pct']}% of all cells"),
                 ("Flagged columns", ov["flagged_columns"], f"of {ov['columns']}")]
        for idx, (a, b, sub) in enumerate(items):
            with c[idx % len(c)]:
                ui.kpi(a, b, sub)
                if compact: st.write("")
    rows = []
    for name, i in prof["columns"].items():
        iss = issues_for(i)
        rows.append({"column": name, "type": KIND_LABEL.get(i["kind"], i["kind"]), "missing %": i["missing_pct"],
                     "unique": i["unique"], "issues": " · ".join(t for t, _ in iss) or "none"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100,
                                                                              format="%.1f%%"),
                                "issues": st.column_config.TextColumn("issues", width="large")})
    return prof


def render():
    ui.page_header("Quality report", "How healthy is the data? Every problem found is explained in plain language.")
    if not state.require_data():
        return
    if st.session_state.steps:
        after = state.current_profile()["overview"]["quality_score"]
        st.info(f"This report describes the file as you uploaded it. After {len(st.session_state.steps)} cleaning steps "
                f"the score is {after:g}/100. Compare before and after on the Clean page.")
    prof = render_profile(st.session_state.raw_df, "", state.steps_key([]))
    st.subheader("What needs attention")
    flagged = [(n, issues_for(i)) for n, i in prof["columns"].items() if issues_for(i)]
    ov = prof["overview"]
    if ov["duplicate_rows"]:
        st.markdown(ui.chip(f"{ov['duplicate_rows']} duplicate rows", "bad"), unsafe_allow_html=True)
    if not flagged and not ov["duplicate_rows"]:
        st.success("No quality problems found. Head to Explore.")
    for name, iss in flagged:
        a, b = st.columns([1, 3])
        a.markdown(f"**{name}**")
        b.markdown("".join(ui.chip(t, k) for t, k in iss), unsafe_allow_html=True)
    st.info("Go to **Clean** to review one-click fixes for these problems.")
