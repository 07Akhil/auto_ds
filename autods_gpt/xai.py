from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def feature_importance(run: Any) -> pd.DataFrame:
    if not run.best_pipeline:
        return pd.DataFrame(columns=["feature", "importance"])

    model = run.best_pipeline.named_steps.get("model")
    names = run.feature_names or []

    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or len(values) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    size = min(len(names), len(values))
    frame = pd.DataFrame({"feature": names[:size], "importance": values[:size]})
    frame["importance"] = frame["importance"].astype(float)
    return frame.sort_values("importance", ascending=False).head(20)


def shap_summary(run: Any) -> tuple[pd.DataFrame, str]:
    if not run.best_pipeline or run.X_test is None or run.X_test.empty:
        return pd.DataFrame(), "SHAP is unavailable because no fitted model and test data are present."
    try:
        import shap

        transformed = run.best_pipeline.named_steps["preprocessor"].transform(run.X_test.head(100))
        model = run.best_pipeline.named_steps["model"]
        explainer = shap.Explainer(model, transformed)
        values = explainer(transformed)
        mean_abs = np.abs(values.values).mean(axis=0)
        names = run.feature_names[: len(mean_abs)]
        summary = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).head(20)
        return summary, "SHAP values show how strongly each transformed feature influenced predictions."
    except Exception as exc:
        return pd.DataFrame(), f"SHAP could not be generated for this model: {exc}"


def lime_status() -> str:
    try:
        import lime  # noqa: F401

        return "LIME is installed and can be used for local row-level explanations."
    except Exception:
        return "LIME is optional. Install lime to enable local row-level explanations."


def root_cause_summary(importances: pd.DataFrame, metrics: pd.DataFrame) -> str:
    if importances.empty:
        return "Root-cause analysis needs a fitted model with feature importance support."
    top_features = ", ".join(importances["feature"].head(5).astype(str).tolist())
    failed = metrics[metrics.get("status", "") != "trained"] if "status" in metrics else pd.DataFrame()
    failure_note = f" {len(failed)} candidate model(s) failed during training and should be reviewed." if not failed.empty else ""
    return f"The strongest drivers are {top_features}. Investigate these fields for data drift, process changes, sensor noise, or business policy effects.{failure_note}"
