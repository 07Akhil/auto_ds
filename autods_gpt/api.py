from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from autods_gpt.automl import run_automl
from autods_gpt.data_intelligence import detect_problem_type, load_dataset, load_sql, profile_dataset
from autods_gpt.llm import GPTReasoner
from autods_gpt.xai import feature_importance, root_cause_summary


app = FastAPI(title="AutoDS-GPT API", version="1.0.0")
reasoner = GPTReasoner()


class SQLRequest(BaseModel):
    connection_uri: str
    query: str
    intent: str = ""
    target: str | None = None


class ChatRequest(BaseModel):
    question: str
    context: dict[str, Any] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "AutoDS-GPT"}


@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    intent: str = Form(""),
    target: str | None = Form(None),
) -> dict[str, Any]:
    raw = await file.read()
    bundle = load_dataset(raw, file.filename)
    return analyze_frame(bundle.frame, intent, target)


@app.post("/analyze/sql")
def analyze_sql(request: SQLRequest) -> dict[str, Any]:
    bundle = load_sql(request.connection_uri, request.query)
    return analyze_frame(bundle.frame, request.intent, request.target)


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    return {"answer": reasoner.chat(request.question, request.context)}


@app.post("/iot/ingest")
def ingest_iot_event(event: dict[str, Any]) -> dict[str, Any]:
    path = Path(tempfile.gettempdir()) / "autods_gpt_iot_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(pd.Series(event).to_json() + "\n")
    return {"status": "accepted", "message": "IoT event queued for stream analytics", "buffer": str(path)}


def analyze_frame(frame: pd.DataFrame, intent: str, target: str | None) -> dict[str, Any]:
    profile = profile_dataset(frame, target)
    problem_type = detect_problem_type(frame, profile, intent)
    run = run_automl(frame, problem_type, profile.get("target_variable"))
    importances = feature_importance(run)
    result_context = {
        "best_model": run.best_model_name,
        "metrics": run.metrics.to_dict(orient="records"),
        "feature_importance": importances.head(10).to_dict(orient="records"),
    }
    return {
        "profile": profile,
        "problem_type": problem_type,
        "dataset_understanding": reasoner.dataset_understanding(profile, intent),
        "problem_explanation": reasoner.problem_detection(profile, intent, problem_type),
        "workflow": reasoner.workflow_generation(profile, intent, problem_type),
        "model_recommendation": reasoner.model_recommendation(profile, problem_type, run.metrics.get("model", pd.Series(dtype=str)).dropna().tolist()),
        "performance_interpretation": reasoner.performance_interpretation(result_context, problem_type),
        "root_cause": root_cause_summary(importances, run.metrics),
        "metrics": run.metrics.to_dict(orient="records"),
        "predictions": run.predictions.head(100).to_dict(orient="records"),
        "feature_importance": importances.to_dict(orient="records"),
    }
