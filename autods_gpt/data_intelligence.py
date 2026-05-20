from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


TEXT_UNIQUE_RATIO = 0.5
HIGH_CARDINALITY_LIMIT = 50


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    source: str


def load_dataset(uploaded_file: Any, source_name: str | None = None) -> DatasetBundle:
    name = source_name or getattr(uploaded_file, "name", "uploaded_dataset")
    suffix = name.lower().split(".")[-1]
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file

    if suffix == "csv":
        frame = pd.read_csv(BytesIO(raw) if isinstance(raw, bytes) else raw)
    elif suffix in {"xls", "xlsx"}:
        frame = pd.read_excel(BytesIO(raw) if isinstance(raw, bytes) else raw)
    elif suffix == "json":
        frame = pd.read_json(BytesIO(raw) if isinstance(raw, bytes) else raw)
    else:
        raise ValueError("Unsupported file type. Use CSV, Excel, or JSON.")

    return DatasetBundle(frame=clean_column_names(frame), source=name)


def load_sql(connection_uri: str, query: str) -> DatasetBundle:
    engine = create_engine(connection_uri)
    frame = pd.read_sql_query(query, engine)
    return DatasetBundle(frame=clean_column_names(frame), source="sql")


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(col).strip().replace(" ", "_") for col in cleaned.columns]
    return cleaned


def profile_dataset(frame: pd.DataFrame, user_target: str | None = None) -> dict[str, Any]:
    sample = frame.head(5).replace({np.nan: None}).to_dict(orient="records")
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = detect_datetime_columns(frame)
    categorical_cols = detect_categorical_columns(frame, numeric_cols, datetime_cols)
    text_cols = detect_text_columns(frame, categorical_cols, datetime_cols)
    target = user_target if user_target in frame.columns else infer_target(frame, numeric_cols, categorical_cols)
    missing = frame.isna().sum().sort_values(ascending=False)
    outliers = detect_outliers(frame[numeric_cols]) if numeric_cols else {}
    correlations = detect_correlations(frame[numeric_cols]) if len(numeric_cols) > 1 else []
    imbalance = detect_imbalance(frame[target]) if target else None

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": frame.columns.tolist(),
        "sample_rows": sample,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_columns": text_cols,
        "datetime_columns": datetime_cols,
        "missing_values": missing[missing > 0].astype(int).to_dict(),
        "missing_value_rate": float(frame.isna().sum().sum() / max(frame.size, 1)),
        "outliers": outliers,
        "high_correlations": correlations,
        "target_variable": target,
        "imbalance": imbalance,
        "high_dimensional": bool(frame.shape[1] > max(100, frame.shape[0] * 0.5)),
        "memory_mb": round(float(frame.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
    }


def detect_datetime_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            cols.append(col)
            continue
        if frame[col].dtype == object and should_attempt_datetime(frame[col], col):
            parsed = pd.to_datetime(frame[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                cols.append(col)
    return cols


def should_attempt_datetime(series: pd.Series, column_name: str) -> bool:
    lower = str(column_name).lower()
    if any(token in lower for token in ["date", "time", "timestamp", "day", "month", "year"]):
        return True
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    has_date_shape = sample.str.contains(r"\d{1,4}[-/:]\d{1,2}", regex=True).mean()
    return bool(has_date_shape > 0.6)


def detect_categorical_columns(frame: pd.DataFrame, numeric_cols: list[str], datetime_cols: list[str]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in datetime_cols:
            continue
        unique_count = frame[col].nunique(dropna=True)
        if col not in numeric_cols and unique_count <= HIGH_CARDINALITY_LIMIT:
            cols.append(col)
        elif col in numeric_cols and unique_count <= 12:
            cols.append(col)
    return cols


def detect_text_columns(frame: pd.DataFrame, categorical_cols: list[str], datetime_cols: list[str]) -> list[str]:
    cols: list[str] = []
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        if col in categorical_cols or col in datetime_cols:
            continue
        avg_len = frame[col].dropna().astype(str).str.len().mean()
        unique_ratio = frame[col].nunique(dropna=True) / max(frame[col].notna().sum(), 1)
        if avg_len and avg_len > 25 and unique_ratio > TEXT_UNIQUE_RATIO:
            cols.append(col)
    return cols


def infer_target(frame: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in frame.columns}
    target_keywords = [
        "target",
        "label",
        "class",
        "outcome",
        "price",
        "sales",
        "revenue",
        "failure",
        "fault",
        "churn",
        "fraud",
        "demand",
        "energy",
        "quality",
    ]
    for keyword in target_keywords:
        for lower, original in lower_map.items():
            if keyword in lower:
                return original
    if frame.columns.size:
        last = frame.columns[-1]
        if last in numeric_cols or last in categorical_cols:
            return last
    return None


def detect_outliers(frame: pd.DataFrame) -> dict[str, int]:
    outliers: dict[str, int] = {}
    for col in frame.columns:
        values = frame[col].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        if mask.any():
            outliers[col] = int(mask.sum())
    return outliers


def detect_correlations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    corr = frame.corr(numeric_only=True).abs()
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= 0.85:
                pairs.append({"left": left, "right": right, "correlation": round(float(value), 3)})
    return pairs[:20]


def detect_imbalance(series: pd.Series) -> dict[str, Any] | None:
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        return None
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return None
    majority_rate = float(counts.iloc[0] / counts.sum())
    return {
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "majority_rate": round(majority_rate, 3),
        "is_imbalanced": majority_rate >= 0.7 and len(counts) > 1,
    }


def detect_problem_type(frame: pd.DataFrame, profile: dict[str, Any], intent: str = "") -> str:
    intent_lower = intent.lower()
    if any(term in intent_lower for term in ["forecast", "time series", "future demand", "energy consumption"]):
        return "time_series_forecasting"
    if any(term in intent_lower for term in ["anomaly", "outlier", "abnormal"]):
        return "anomaly_detection"
    if any(term in intent_lower for term in ["segment", "cluster", "group customers"]):
        return "clustering"

    target = profile.get("target_variable")
    if not target:
        return "clustering"

    if profile.get("datetime_columns") and target in profile.get("numeric_columns", []) and frame.shape[0] >= 20:
        if any(term in str(target).lower() for term in ["demand", "sales", "energy", "load", "usage"]):
            return "time_series_forecasting"

    target_series = frame[target]
    unique_count = target_series.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(target_series) and unique_count > 20:
        return "regression"
    return "classification"
