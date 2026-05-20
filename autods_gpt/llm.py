import os
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


SYSTEM_PROMPT = "You are an expert AI Data Scientist and AutoML assistant."


@dataclass
class LLMConfig:
    api_key: str | None = None
    base_url: str = "https://api.hpc-ai.com/inference/v1"
    model: str = "openai/gpt-5.5"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("AUTODS_GPT_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("AUTODS_GPT_BASE_URL", "https://api.hpc-ai.com/inference/v1"),
            model=os.getenv("AUTODS_GPT_MODEL", "openai/gpt-5.5"),
        )


class GPTReasoner:
    """OpenAI-compatible GPT reasoning layer for AutoDS-GPT."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self.enabled = bool(self.config.api_key and OpenAI)
        self.client = None
        if self.enabled:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

    def complete(self, prompt: str, fallback: str) -> str:
        if not self.enabled:
            return fallback

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or fallback
        except Exception as exc:  # API failures should not break local AutoML.
            return f"{fallback}\n\nLLM note: GPT reasoning was unavailable ({exc})."

    def dataset_understanding(self, profile: dict[str, Any], intent: str) -> str:
        prompt = f"""
Analyze this dataset profile and business intent for an enterprise AutoML platform.

Business intent:
{intent or "Not provided"}

Dataset profile:
{profile}

Explain the likely dataset purpose, important columns, risks, possible use cases,
and recommended next actions in clear business language.
"""
        return self.complete(prompt, fallback_dataset_understanding(profile, intent))

    def problem_detection(self, profile: dict[str, Any], intent: str, detected_problem: str) -> str:
        prompt = f"""
The local AutoML engine detected problem type: {detected_problem}.

Business intent:
{intent or "Not provided"}

Dataset profile:
{profile}

Explain why this problem type is appropriate. Mention the target variable, data
shape, and any uncertainty. If another problem type could also fit, explain it.
"""
        return self.complete(prompt, f"The selected problem type is {detected_problem} based on target and column patterns.")

    def workflow_generation(self, profile: dict[str, Any], intent: str, detected_problem: str) -> str:
        prompt = f"""
Generate a production-oriented machine learning workflow for this AutoDS-GPT run.

Problem type: {detected_problem}
Business intent: {intent or "Not provided"}
Dataset profile: {profile}

Include preprocessing, feature engineering, candidate models, validation,
evaluation metrics, XAI, deployment, monitoring, and industrial operations steps.
"""
        return self.complete(prompt, fallback_workflow(detected_problem))

    def model_recommendation(self, profile: dict[str, Any], problem_type: str, candidates: list[str]) -> str:
        prompt = f"""
Recommend models for this dataset and explain why.

Problem type: {problem_type}
Candidate models trained or considered: {candidates}
Dataset profile: {profile}

For each model, explain advantages, disadvantages, expected performance, and
computational complexity in concise enterprise language.
"""
        return self.complete(prompt, fallback_model_recommendation(problem_type, candidates))

    def performance_interpretation(self, results: dict[str, Any], problem_type: str) -> str:
        prompt = f"""
Interpret these AutoML results for a business user.

Problem type: {problem_type}
Results: {results}

Explain model quality, risks, operational meaning, possible root causes of poor
performance, and how to improve the result.
"""
        return self.complete(prompt, fallback_performance_interpretation(results))

    def chat(self, question: str, context: dict[str, Any]) -> str:
        prompt = f"""
You are AutoDS-GPT, a conversational analytics assistant.

Context:
{context}

User question:
{question}

Answer clearly. Use the dataset, workflow, metrics, and XAI context when relevant.
"""
        return self.complete(prompt, "I can help interpret the dataset, model results, feature importance, failures, and business insights once analysis has run.")


def fallback_dataset_understanding(profile: dict[str, Any], intent: str) -> str:
    target = profile.get("target_variable") or "not yet identified"
    return (
        f"This dataset has {profile.get('rows')} rows and {profile.get('columns')} columns. "
        f"The likely target variable is {target}. "
        f"Business intent: {intent or 'general automated data science analysis'}. "
        "AutoDS-GPT will profile data quality, prepare features, train candidate models, and generate explanations."
    )


def fallback_workflow(problem_type: str) -> str:
    return (
        f"Workflow: ingest data, profile schema, clean missing values, encode categorical fields, "
        f"scale numerical fields when useful, train candidate {problem_type} models, compare validation metrics, "
        "explain the best model with feature importance/XAI, generate business insights, and prepare deployment artifacts."
    )


def fallback_model_recommendation(problem_type: str, candidates: list[str]) -> str:
    return f"Recommended {problem_type} candidates: {', '.join(candidates) or 'baseline Scikit-learn models'}."


def fallback_performance_interpretation(results: dict[str, Any]) -> str:
    best = results.get("best_model", "the top-ranked model")
    return f"{best} performed best in the current run. Review the metric table and feature importance before production use."
