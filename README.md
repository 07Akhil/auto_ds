# AutoDS-GPT

AutoDS-GPT is an enterprise-style AI data science platform that combines LLM reasoning, automated machine learning, explainable AI, and adaptive workflow generation for business and industrial analytics.

It supports uploaded datasets, automatic dataset profiling, problem type detection, model recommendation, dynamic pipeline training, XAI summaries, downloadable reports, and a conversational AI assistant powered by an OpenAI-compatible HPC-AI endpoint.

## Core Stack

- Backend API: FastAPI
- Dashboard: Streamlit
- ML: Pandas, NumPy, Scikit-learn
- Optional ML/XAI: XGBoost, LightGBM, SHAP, LIME
- Visualization: Plotly, Matplotlib
- LLM: `openai/gpt-5.5` via `https://api.hpc-ai.com/inference/v1`
- Deployment: Docker

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:AUTODS_GPT_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

Run the API separately:

```powershell
uvicorn autods_gpt.api:app --host 0.0.0.0 --port 8000
```

## Environment Variables

```text
AUTODS_GPT_API_KEY=YOUR_API_KEY
AUTODS_GPT_BASE_URL=https://api.hpc-ai.com/inference/v1
AUTODS_GPT_MODEL=openai/gpt-5.5
```

## What It Does

- Detects numerical, categorical, text, datetime, and target columns.
- Identifies missing values, outliers, correlations, imbalance, and time-series signals.
- Uses GPT reasoning for dataset understanding, business intent interpretation, workflow generation, model recommendation, insight generation, and conversational analytics.
- Trains regression, classification, clustering, and anomaly detection pipelines.
- Produces feature importance, confusion matrices, forecast-ready plots, model comparison tables, and beginner-friendly explanations.
- Exposes REST endpoints for enterprise integration.

## Notes

Heavy optional models such as TensorFlow LSTM and Prophet are represented as recommendation targets in the AI workflow layer. The default local training path focuses on reliable Scikit-learn pipelines, with optional XGBoost and LightGBM when installed.
