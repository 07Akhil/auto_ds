from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def build_text_report(profile: dict[str, Any], problem_type: str, metrics: pd.DataFrame, insights: str, workflow: str) -> str:
    metrics_text = metrics.to_string(index=False) if not metrics.empty else "No model metrics available."
    return f"""AutoDS-GPT Analysis Report

Dataset
- Rows: {profile.get("rows")}
- Columns: {profile.get("columns")}
- Target: {profile.get("target_variable")}
- Problem type: {problem_type}
- Missing value rate: {profile.get("missing_value_rate")}

Model Metrics
{metrics_text}

AI Insights
{insights}

Generated Workflow
{workflow}
"""


def build_pdf_report(profile: dict[str, Any], problem_type: str, metrics: pd.DataFrame, insights: str, workflow: str) -> bytes:
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AutoDS-GPT Analysis Report", ln=True)
        pdf.set_font("Helvetica", size=10)
        text = build_text_report(profile, problem_type, metrics, insights, workflow)
        for line in text.splitlines():
            pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))
        return bytes(pdf.output(dest="S"))
    except Exception:
        return build_text_report(profile, problem_type, metrics, insights, workflow).encode("utf-8")


def dataframe_download(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    frame.to_csv(output, index=False)
    return output.getvalue()
