"""📤 Upload Dataset page.

Implements the full workflow from the spec:
STEP 1 Upload -> STEP 2 Preview/Profile -> STEP 3 Data Quality ->
STEP 4 Column Mapping -> STEP 5 Target Detection -> Train (Type A) or
Score-only / Segmentation (Type B).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, DEFAULT_MODEL_METADATA_PATH, DEFAULT_MODEL_PATH, PRIVACY_NOTICE, TARGET_COLUMN
from src.data.column_mapper import suggest_mapping_for_dataset
from src.data.leakage_detector import detect_leakage
from src.data.loader import DatasetLoadError, load_dataset
from src.data.pii_detector import detect_pii_and_sensitive_columns
from src.data.profiler import class_balance_report, profile_dataset
from src.data_adapters.generic_adapter import GenericAdapter
from src.modeling.pipeline_builder import AVAILABLE_MODELS
from src.modeling.predict import predict_batch
from src.modeling.train import TrainingError, train_model
from src.ui.state import get_active_pipeline, set_user_trained_model


def render() -> None:
    st.title("📤 Upload Dataset")
    st.info(PRIVACY_NOTICE)

    uploaded_file = st.file_uploader("Upload a CSV or Excel (.xlsx) file", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.caption("No file uploaded yet. Upload a dataset to begin.")
        return

    try:
        load_result = load_dataset(uploaded_file, uploaded_file.name)
    except DatasetLoadError as exc:
        st.error(str(exc))
        return

    for w in load_result.warnings:
        st.warning(w)

    raw_df = load_result.dataframe
    st.session_state["raw_upload_df"] = raw_df

    # ---- STEP 2: Preview & profile ----------------------------------------------------
    st.divider()
    st.header("Step 1 — Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{load_result.n_rows:,}")
    c2.metric("Columns", load_result.n_cols)

    profile = profile_dataset(raw_df)
    c3.metric("Duplicate Rows", profile.n_duplicate_rows)

    with st.expander("📈 Descriptive statistics & column types", expanded=False):
        st.write("**Numeric columns:**", profile.numeric_columns or "None detected")
        if not profile.describe_numeric.empty:
            st.dataframe(profile.describe_numeric, use_container_width=True)
        st.write("**Categorical columns:**", profile.categorical_columns or "None detected")
        dtype_table = pd.DataFrame(
            [{"Column": c.name, "Dtype": c.dtype, "Inferred Type": c.inferred_type,
              "Missing": c.n_missing, "% Missing": f"{c.pct_missing:.1%}", "Unique": c.n_unique}
             for c in profile.columns]
        )
        st.dataframe(dtype_table, use_container_width=True, hide_index=True)

    if profile.warnings:
        st.subheader("⚠️ Data Quality Warnings")
        for w in profile.warnings:
            st.warning(w)

    # ---- STEP 3: PII & leakage detection ------------------------------------------------
    st.divider()
    st.header("Step 2 — PII, Sensitive Attributes & Leakage Checks")
    pii_flags = detect_pii_and_sensitive_columns(raw_df)
    leakage_flags = detect_leakage(raw_df)

    recommended_exclusions = set()
    if pii_flags:
        st.warning("Potential PII / sensitive-attribute columns detected:")
        for f in pii_flags:
            st.markdown(f"- **{f.column}** ({f.kind}): {f.reason}")
            recommended_exclusions.add(f.column)
    else:
        st.success("No obvious PII or sensitive-attribute columns detected by name/pattern matching.")

    if leakage_flags:
        st.warning("Potential target-leakage columns detected:")
        for f in leakage_flags:
            st.markdown(f"- **{f.column}**: {f.reason}")
            recommended_exclusions.add(f.column)
    else:
        st.success("No obvious target-leakage columns detected by keyword matching.")

    # ---- STEP 4: Column mapping ----------------------------------------------------------
    st.divider()
    st.header("Step 3 — Column Mapping")
    st.caption(
        "The system suggests mappings automatically, but you always have final control. "
        "Set a column to '(ignore)' to exclude it entirely."
    )

    suggestions = suggest_mapping_for_dataset(list(raw_df.columns))
    canonical_options = ["(ignore)"] + sorted({s.suggested_target for s in suggestions if s.suggested_target} |
                                               {"annual_income", "monthly_income", "employment_type",
                                                "employment_duration_years", "loan_amount", "loan_term_months",
                                                "interest_rate", "loan_purpose", "residential_status",
                                                "existing_debt", "existing_emi", "credit_history_years",
                                                "credit_utilization", "total_credit_accounts",
                                                "previous_delinquencies", TARGET_COLUMN, "customer_id"})

    mapping_df = pd.DataFrame([
        {
            "Original Column": s.source_column,
            "Suggested Mapping": s.suggested_target or "(ignore)",
            "Confidence": s.confidence,
            "Method": s.method,
            "User Selected Mapping": s.suggested_target or "(ignore)",
        }
        for s in suggestions
    ])

    edited = st.data_editor(
        mapping_df,
        column_config={
            "User Selected Mapping": st.column_config.SelectboxColumn(options=canonical_options, required=True),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1),
        },
        disabled=["Original Column", "Suggested Mapping", "Confidence", "Method"],
        use_container_width=True, hide_index=True, key="mapping_editor",
    )

    confirm_mapping = st.button("✅ Confirm Column Mapping", type="primary")
    if confirm_mapping:
        mapping = {
            row["Original Column"]: row["User Selected Mapping"]
            for _, row in edited.iterrows() if row["User Selected Mapping"] != "(ignore)"
        }
        st.session_state["column_mapping"] = mapping
        st.session_state["excluded_columns"] = list(recommended_exclusions)

    mapping = st.session_state.get("column_mapping")
    if not mapping:
        st.info("Confirm the column mapping above to continue.")
        return

    canonical_df = GenericAdapter.transform_with_mapping(raw_df, mapping)
    st.success(f"Mapping applied. Canonical dataset has {canonical_df.shape[1]} mapped column(s).")
    st.dataframe(canonical_df.head(10), use_container_width=True)

    # ---- STEP 5: Target detection ---------------------------------------------------------
    st.divider()
    st.header("Step 4 — Target Detection")
    has_target = TARGET_COLUMN in canonical_df.columns

    if has_target:
        st.success(f"A target/outcome column was mapped to `{TARGET_COLUMN}`.")
        raw_target_values = canonical_df[TARGET_COLUMN].dropna().unique().tolist()
        st.write("Observed values:", raw_target_values[:10])

        if set(pd.Series(raw_target_values).astype(str)) - {"0", "1"}:
            default_values = st.multiselect(
                "Select which value(s) represent DEFAULT (=1). All other values become 0 (no default).",
                options=raw_target_values,
            )
            canonical_df[TARGET_COLUMN] = canonical_df[TARGET_COLUMN].apply(
                lambda v: 1 if v in default_values else (0 if pd.notna(v) else None)
            )
        else:
            canonical_df[TARGET_COLUMN] = pd.to_numeric(canonical_df[TARGET_COLUMN], errors="coerce")

        _render_type_a_training(canonical_df, recommended_exclusions)
    else:
        st.warning(
            "This dataset does not contain an observed repayment/default outcome, so a "
            "supervised credit-default model cannot be trained from it."
        )
        _render_type_b_options(canonical_df)


def _render_type_a_training(canonical_df: pd.DataFrame, recommended_exclusions: set) -> None:
    st.divider()
    st.header("Step 5 — Train a Model on This Dataset")

    balance = class_balance_report(canonical_df, TARGET_COLUMN)
    for w in balance["warnings"]:
        st.warning(w)
    st.write("**Class distribution:**", balance["counts"])

    col1, col2, col3 = st.columns(3)
    model_name = col1.selectbox("Model", options=list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS).index("Random Forest") if "Random Forest" in AVAILABLE_MODELS else 0)
    threshold_strategy = col2.selectbox("Threshold optimization", options=["f1", "recall", "precision"], index=0)
    test_size = col3.slider("Validation split", 0.1, 0.4, 0.2, 0.05)

    exclude_options = [c for c in canonical_df.columns if c != TARGET_COLUMN]
    excluded = st.multiselect(
        "Columns to exclude from training (leakage/PII flagged columns are pre-selected if present)",
        options=exclude_options,
        default=[c for c in exclude_options if c in recommended_exclusions],
    )

    if st.button("🚀 Train Model", type="primary"):
        try:
            with st.spinner("Training model..."):
                result = train_model(
                    canonical_df, target_col=TARGET_COLUMN, model_name=model_name,
                    threshold_strategy=threshold_strategy, test_size=test_size,
                    excluded_columns=excluded, data_source_type="user_uploaded",
                )
        except TrainingError as exc:
            st.error(str(exc))
            return

        for w in result.warnings:
            st.warning(w)

        st.success("Model trained successfully.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC", f"{result.evaluation.roc_auc:.3f}")
        m2.metric("Accuracy", f"{result.evaluation.accuracy:.3f}")
        m3.metric("Precision", f"{result.evaluation.precision:.3f}")
        m4.metric("Recall", f"{result.evaluation.recall:.3f}")
        m5.metric("F1", f"{result.evaluation.f1:.3f}")

        st.write("**Confusion Matrix** (rows = actual, cols = predicted)")
        st.dataframe(pd.DataFrame(result.evaluation.confusion,
                                   index=["Actual: No Default", "Actual: Default"],
                                   columns=["Pred: No Default", "Pred: Default"]))

        st.session_state["_pending_trained_model"] = result

    pending = st.session_state.get("_pending_trained_model")
    if pending is not None:
        if st.button("✅ Set as Active Model for this Session", type="primary"):
            set_user_trained_model(pending.pipeline, pending.metadata)
            st.session_state["active_dataset_df"] = canonical_df
            st.session_state["active_dataset_source"] = "uploaded"
            st.session_state["active_dataset_name"] = "Uploaded dataset (user-trained model)"
            st.session_state["active_dataset_has_target"] = True
            st.session_state["active_dataset_target_col"] = TARGET_COLUMN
            st.success("This model and dataset are now active across the application.")


def _render_type_b_options(canonical_df: pd.DataFrame) -> None:
    st.subheader("Options")
    choice = st.radio(
        "What would you like to do with this dataset?",
        options=[
            "Score customers using the application's pre-trained default model",
            "Customer segmentation / risk-proxy analysis (NOT default predictions)",
        ],
    )

    if choice.startswith("Score"):
        if st.button("📊 Score with pre-trained model & set as active dataset", type="primary"):
            st.session_state["active_dataset_df"] = canonical_df
            st.session_state["active_dataset_source"] = "uploaded"
            st.session_state["active_dataset_name"] = "Uploaded dataset (scored with built-in model)"
            st.session_state["active_dataset_has_target"] = False
            st.session_state["active_dataset_target_col"] = None
            st.session_state["scored_df"] = None
            from src.modeling.train import load_metadata, load_model

            st.session_state["active_pipeline"] = load_model(DEFAULT_MODEL_PATH)
            st.session_state["active_model_metadata"] = load_metadata(DEFAULT_MODEL_METADATA_PATH)
            st.session_state["active_model_source"] = "builtin"
            st.success("Uploaded dataset is now active and scored with the built-in demonstration model. "
                       "Visit Portfolio Analytics or Customer Risk Assessment.")
    else:
        st.info(
            "This performs **descriptive segmentation and outlier analysis only**. "
            "These are NOT default predictions, since no observed outcome exists in this data."
        )
        _render_segmentation(canonical_df)


def _render_segmentation(canonical_df: pd.DataFrame) -> None:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from src.config import NUMERIC_RAW_FEATURES

    numeric_cols = [c for c in NUMERIC_RAW_FEATURES if c in canonical_df.columns]
    if len(numeric_cols) < 2:
        st.warning("Not enough recognized numeric columns for segmentation.")
        return

    n_clusters = st.slider("Number of segments", 2, 6, 3)
    X = SimpleImputer(strategy="median").fit_transform(canonical_df[numeric_cols])
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)

    result_df = canonical_df.copy()
    result_df["segment"] = labels

    st.write("**Segment sizes:**")
    st.bar_chart(result_df["segment"].value_counts().sort_index())

    st.write("**Segment profile (mean of numeric fields):**")
    st.dataframe(result_df.groupby("segment")[numeric_cols].mean().round(2), use_container_width=True)

    # Simple outlier flag: > 3 std dev on any numeric feature
    import numpy as np

    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9))
    outlier_flag = (z_scores > 3).any(axis=1)
    st.write(f"**Outliers detected:** {int(outlier_flag.sum())} of {len(result_df)} rows (|z| > 3 on any numeric field).")
