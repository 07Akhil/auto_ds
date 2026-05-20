from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from autods_gpt.data_intelligence import should_attempt_datetime

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None


@dataclass
class AutoMLRun:
    problem_type: str
    target: str | None
    best_model_name: str | None
    best_pipeline: Any
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion: np.ndarray | None
    feature_names: list[str]
    X_train: pd.DataFrame | None
    X_test: pd.DataFrame | None
    y_train: pd.Series | None
    y_test: pd.Series | None


def build_preprocessor(frame: pd.DataFrame, target: str | None) -> tuple[ColumnTransformer, list[str], list[str]]:
    features = frame.drop(columns=[target], errors="ignore") if target else frame.copy()
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in features.columns if col not in numeric]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", max_categories=30)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return preprocessor, numeric, categorical


def candidate_models(problem_type: str) -> dict[str, Any]:
    if problem_type == "regression" or problem_type == "time_series_forecasting":
        models: dict[str, Any] = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1),
            "SVR": SVR(),
        }
        if XGBRegressor:
            models["XGBoost Regressor"] = XGBRegressor(n_estimators=120, random_state=42, objective="reg:squarederror")
        if LGBMRegressor:
            models["LightGBM Regressor"] = LGBMRegressor(n_estimators=160, random_state=42, verbose=-1)
        return models

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=-1),
            "SVM Classifier": SVC(probability=True),
        }
        if XGBClassifier:
            models["XGBoost Classifier"] = XGBClassifier(n_estimators=120, random_state=42, eval_metric="logloss")
        if LGBMClassifier:
            models["LightGBM Classifier"] = LGBMClassifier(n_estimators=160, random_state=42, verbose=-1)
        return models

    if problem_type == "anomaly_detection":
        return {"Isolation Forest": IsolationForest(contamination="auto", random_state=42)}

    return {
        "K-Means": KMeans(n_clusters=3, random_state=42, n_init="auto"),
        "DBSCAN": DBSCAN(eps=1.2, min_samples=5),
    }


def run_automl(frame: pd.DataFrame, problem_type: str, target: str | None, expert_options: dict[str, Any] | None = None) -> AutoMLRun:
    options = expert_options or {}
    working = prepare_frame(frame)
    preprocessor, _, _ = build_preprocessor(working, target)

    if problem_type in {"clustering", "anomaly_detection"} or not target:
        return run_unsupervised(working, preprocessor, problem_type)

    X = working.drop(columns=[target])
    y = working[target]
    stratify = y if problem_type == "classification" and y.nunique() <= 20 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=stratify)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

    rows: list[dict[str, Any]] = []
    best_name: str | None = None
    best_pipeline = None
    best_score = -np.inf
    best_predictions: np.ndarray | None = None
    confusion = None

    for name, model in candidate_models(problem_type).items():
        if options.get("model_allowlist") and name not in options["model_allowlist"]:
            continue
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        try:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            row = score_model(name, problem_type, y_test, pred, pipeline, X_train, y_train)
            rows.append(row)
            ranking_score = row.get("r2", row.get("f1", row.get("accuracy", -row.get("rmse", 0))))
            if ranking_score > best_score:
                best_score = float(ranking_score)
                best_name = name
                best_pipeline = pipeline
                best_predictions = pred
                if problem_type == "classification":
                    confusion = confusion_matrix(y_test, pred)
        except Exception as exc:
            rows.append({"model": name, "status": f"failed: {exc}"})

    predictions = pd.DataFrame({"actual": y_test.reset_index(drop=True)})
    if best_predictions is not None:
        predictions["predicted"] = best_predictions

    return AutoMLRun(
        problem_type=problem_type,
        target=target,
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        metrics=pd.DataFrame(rows),
        predictions=predictions,
        confusion=confusion,
        feature_names=get_feature_names(best_pipeline) if best_pipeline else [],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def run_unsupervised(frame: pd.DataFrame, preprocessor: ColumnTransformer, problem_type: str) -> AutoMLRun:
    X = frame.copy()
    rows: list[dict[str, Any]] = []
    best_name = None
    best_pipeline = None
    best_labels = None
    best_score = -np.inf

    for name, model in candidate_models(problem_type).items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        try:
            labels = pipeline.fit_predict(X) if hasattr(pipeline, "fit_predict") else pipeline.fit(X).predict(X)
            unique_labels = len(set(labels))
            score = None
            transformed = pipeline.named_steps["preprocessor"].transform(X)
            if unique_labels > 1 and unique_labels < len(X):
                score = silhouette_score(transformed, labels)
            anomaly_rate = float(np.mean(np.array(labels) == -1)) if problem_type == "anomaly_detection" else None
            rows.append(
                {
                    "model": name,
                    "status": "trained",
                    "clusters_or_segments": unique_labels,
                    "silhouette": score,
                    "anomaly_rate": anomaly_rate,
                }
            )
            ranking_score = score if score is not None else 0
            if ranking_score >= best_score:
                best_score = float(ranking_score)
                best_name = name
                best_pipeline = pipeline
                best_labels = labels
        except Exception as exc:
            rows.append({"model": name, "status": f"failed: {exc}"})

    predictions = pd.DataFrame({"segment_or_anomaly": best_labels}) if best_labels is not None else pd.DataFrame()
    return AutoMLRun(
        problem_type=problem_type,
        target=None,
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        metrics=pd.DataFrame(rows),
        predictions=predictions,
        confusion=None,
        feature_names=get_feature_names(best_pipeline) if best_pipeline else [],
        X_train=X,
        X_test=X,
        y_train=None,
        y_test=None,
    )


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for col in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[col]):
            prepared[f"{col}_year"] = prepared[col].dt.year
            prepared[f"{col}_month"] = prepared[col].dt.month
            prepared[f"{col}_day"] = prepared[col].dt.day
            prepared = prepared.drop(columns=[col])
        elif prepared[col].dtype == object and should_attempt_datetime(prepared[col], col):
            parsed = pd.to_datetime(prepared[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                prepared[f"{col}_year"] = parsed.dt.year
                prepared[f"{col}_month"] = parsed.dt.month
                prepared[f"{col}_day"] = parsed.dt.day
                prepared = prepared.drop(columns=[col])
    return prepared


def score_model(name: str, problem_type: str, y_true: pd.Series, pred: np.ndarray, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    row: dict[str, Any] = {"model": name, "status": "trained"}
    if problem_type == "classification":
        row["accuracy"] = round(float(accuracy_score(y_true, pred)), 4)
        row["f1"] = round(float(f1_score(y_true, pred, average="weighted", zero_division=0)), 4)
        scoring = "f1_weighted"
    else:
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        row["rmse"] = round(float(rmse), 4)
        row["mae"] = round(float(mean_absolute_error(y_true, pred)), 4)
        row["r2"] = round(float(r2_score(y_true, pred)), 4)
        scoring = "r2"

    if len(X_train) >= 20:
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=min(5, max(2, len(X_train) // 10)), scoring=scoring)
            row["cross_val_mean"] = round(float(np.mean(scores)), 4)
        except Exception:
            pass
    return row


def get_feature_names(pipeline: Pipeline | None) -> list[str]:
    if not pipeline:
        return []
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []
