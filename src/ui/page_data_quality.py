"""🔍 Data Quality page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data.leakage_detector import detect_leakage
from src.data.pii_detector import detect_pii_and_sensitive_columns
from src.data.profiler import class_balance_report, profile_dataset
from src.ui.state import get_active_dataset


def render() -> None:
    st.title("🔍 Data Quality")

    df = get_active_dataset()
    if df is None:
        st.info("No active dataset available.")
        return

    profile = profile_dataset(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{profile.n_rows:,}")
    c2.metric("Columns", profile.n_cols)
    c3.metric("Duplicate Rows", profile.n_duplicate_rows)

    st.divider()
    st.subheader("Missing Values")
    missing_table = pd.DataFrame(
        [{"Column": c.name, "Missing": c.n_missing, "% Missing": f"{c.pct_missing:.1%}"}
         for c in profile.columns if c.n_missing > 0]
    )
    if missing_table.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Feature Types")
    type_table = pd.DataFrame(
        [{"Column": c.name, "Dtype": c.dtype, "Inferred Type": c.inferred_type, "Unique Values": c.n_unique}
         for c in profile.columns]
    )
    st.dataframe(type_table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Outliers (numeric columns, |z-score| > 3)")
    outlier_rows = []
    for col in profile.numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 10 or series.std() == 0:
            continue
        z = (series - series.mean()) / series.std()
        n_outliers = int((z.abs() > 3).sum())
        if n_outliers > 0:
            outlier_rows.append({"Column": col, "Outlier Count": n_outliers, "% of Rows": f"{n_outliers/len(series):.1%}"})
    if outlier_rows:
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No extreme outliers (|z| > 3) detected in numeric columns.")

    st.divider()
    st.subheader("Class Imbalance")
    target_col = st.session_state.get("active_dataset_target_col")
    if target_col and target_col in df.columns:
        balance = class_balance_report(df, target_col)
        st.write("**Class counts:**", balance["counts"])
        for w in balance["warnings"]:
            st.warning(w)
        if not balance["warnings"]:
            st.success("Target class balance looks reasonable.")
    else:
        st.caption("No target/outcome column is active for this dataset (scoring-only dataset).")

    st.divider()
    st.subheader("Potential Target Leakage")
    leakage_flags = detect_leakage(df, target_col if target_col and target_col in df.columns else None)
    if leakage_flags:
        for f in leakage_flags:
            st.warning(f"**{f.column}**: {f.reason}")
    else:
        st.success("No leakage indicators detected.")

    st.divider()
    st.subheader("Sensitive Attributes / PII")
    pii_flags = detect_pii_and_sensitive_columns(df)
    if pii_flags:
        for f in pii_flags:
            st.warning(f"**{f.column}** ({f.kind}): {f.reason}")
        st.caption("These should be excluded before training a new model (see the Upload Dataset page).")
    else:
        st.success("No obvious PII or sensitive-attribute columns detected.")

    st.divider()
    st.caption(
        "No arbitrary 'data quality score' is shown here because a single opaque number "
        "would hide which specific issues drive it — the checks above are transparent and "
        "individually actionable instead."
    )
