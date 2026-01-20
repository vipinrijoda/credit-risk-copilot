"""Builds the single sklearn Pipeline used for training AND prediction.

Pipeline = FinancialFeatureEngineer -> ColumnTransformer(impute/scale/encode) -> Classifier

Using ONE pipeline object for training, validation, and every prediction
path (built-in dataset, uploaded dataset, manual new-customer form) is what
prevents training/inference skew (see project requirement #39/#11).
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.modeling.feature_engineering import FinancialFeatureEngineer

AVAILABLE_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

# Optional heavyweight models — only registered if installed, so the app
# never hard-fails when XGBoost/LightGBM/CatBoost are unavailable.
try:
    from xgboost import XGBClassifier  # type: ignore

    AVAILABLE_MODELS["XGBoost"] = lambda: XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier  # type: ignore

    AVAILABLE_MODELS["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=300, max_depth=-1, learning_rate=0.05, random_state=42, verbose=-1
    )
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier  # type: ignore

    AVAILABLE_MODELS["CatBoost"] = lambda: CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05, verbose=False, random_state=42
    )
except ImportError:
    pass


@dataclass
class SchemaColumns:
    numeric: list[str]
    categorical: list[str]


def build_preprocessor(schema: SchemaColumns) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if schema.numeric:
        transformers.append(("numeric", numeric_pipe, schema.numeric))
    if schema.categorical:
        transformers.append(("categorical", categorical_pipe, schema.categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(schema: SchemaColumns, model_name: str = "Random Forest") -> Pipeline:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not available. Installed options: {list(AVAILABLE_MODELS)}"
        )

    feature_engineer = FinancialFeatureEngineer()
    preprocessor = build_preprocessor(schema)
    classifier = AVAILABLE_MODELS[model_name]()

    return Pipeline(steps=[
        ("feature_engineering", feature_engineer),
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])
