from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from autods_gpt.automl import run_automl
from autods_gpt.data_intelligence import detect_problem_type, load_dataset, load_sql, profile_dataset
from autods_gpt.llm import GPTReasoner
from autods_gpt.reporting import build_pdf_report, dataframe_download
from autods_gpt.xai import feature_importance, lime_status, root_cause_summary, shap_summary


st.set_page_config(page_title="AutoDS-GPT", page_icon="AI", layout="wide")


def init_state() -> None:
    defaults = {
        "frame": None,
        "profile": None,
        "problem_type": None,
        "run": None,
        "intent": "",
        "ai": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    init_state()
    reasoner = GPTReasoner()

    st.title("AutoDS-GPT")
    st.caption("Enterprise AutoML + AI Copilot for industrial and business data science.")

    with st.sidebar:
        st.header("Workspace")
        mode = st.radio("User mode", ["Beginner Mode", "Expert Mode"], horizontal=False)
        intent = st.text_area(
            "Business intent",
            placeholder="Predict machine failure using sensor data, forecast energy consumption, detect fraud...",
            height=110,
        )
        st.session_state.intent = intent
        source = st.tabs(["Upload", "SQL", "IoT"])

        with source[0]:
            uploaded = st.file_uploader("Dataset", type=["csv", "xlsx", "xls", "json"])
            if uploaded and st.button("Load dataset", use_container_width=True):
                bundle = load_dataset(uploaded, uploaded.name)
                st.session_state.frame = bundle.frame
                st.success(f"Loaded {bundle.source}")

        with source[1]:
            uri = st.text_input("SQLAlchemy connection URI", placeholder="postgresql+psycopg2://user:pass@host/db")
            query = st.text_area("SQL query", placeholder="SELECT * FROM production_events LIMIT 5000")
            if st.button("Load SQL data", use_container_width=True):
                try:
                    bundle = load_sql(uri, query)
                    st.session_state.frame = bundle.frame
                    st.success("Loaded SQL dataset")
                except Exception as exc:
                    st.error(f"SQL load failed: {exc}")

        with source[2]:
            st.info("Real-time IoT support is exposed through the FastAPI `/iot/ingest` endpoint. Batch sensor CSV files can be uploaded here for predictive maintenance and anomaly analysis.")

    if st.session_state.frame is None:
        render_empty_state()
        return

    frame = st.session_state.frame
    target_options = ["Auto detect"] + frame.columns.tolist()

    controls = st.columns([2, 1, 1, 1])
    with controls[0]:
        selected_target = st.selectbox("Target variable", target_options)
    with controls[1]:
        force_problem = st.selectbox("Problem type", ["Auto detect", "Regression", "Classification", "Clustering", "Anomaly Detection", "Time-Series Forecasting"])
    with controls[2]:
        max_rows = st.number_input("Training row limit", min_value=100, max_value=100000, value=min(max(len(frame), 100), 5000), step=100)
    with controls[3]:
        analyze = st.button("Run AI analysis", type="primary", use_container_width=True)

    if analyze:
        with st.spinner("AutoDS-GPT is profiling data, generating a workflow, and training candidate models..."):
            run_analysis(reasoner, frame.head(int(max_rows)), intent, selected_target, force_problem, mode)

    render_dashboard(reasoner)


def run_analysis(reasoner: GPTReasoner, frame: pd.DataFrame, intent: str, selected_target: str, force_problem: str, mode: str) -> None:
    target = None if selected_target == "Auto detect" else selected_target
    profile = profile_dataset(frame, target)
    problem_type = detect_problem_type(frame, profile, intent)
    if force_problem != "Auto detect":
        problem_type = force_problem.lower().replace("-", "_").replace(" ", "_")

    expert_options = {}
    if mode == "Expert Mode":
        expert_options["advanced_controls"] = True

    run = run_automl(frame, problem_type, profile.get("target_variable"), expert_options)
    importances = feature_importance(run)
    results = {
        "best_model": run.best_model_name,
        "metrics": run.metrics.to_dict(orient="records"),
        "feature_importance": importances.head(10).to_dict(orient="records"),
    }

    st.session_state.profile = profile
    st.session_state.problem_type = problem_type
    st.session_state.run = run
    st.session_state.ai = {
        "dataset": reasoner.dataset_understanding(profile, intent),
        "problem": reasoner.problem_detection(profile, intent, problem_type),
        "workflow": reasoner.workflow_generation(profile, intent, problem_type),
        "models": reasoner.model_recommendation(profile, problem_type, run.metrics.get("model", pd.Series(dtype=str)).dropna().tolist()),
        "performance": reasoner.performance_interpretation(results, problem_type),
        "root_cause": root_cause_summary(importances, run.metrics),
    }


def render_empty_state() -> None:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Start with a dataset")
        st.write("Upload CSV, Excel, or JSON data, describe the business goal, then run one-click AI analysis.")
        st.write("AutoDS-GPT will detect schema, infer the ML problem, train candidate pipelines, rank models, explain results, and generate an industrial workflow.")
    with right:
        st.subheader("Industrial use cases")
        st.write("- Predictive maintenance")
        st.write("- IoT sensor anomaly detection")
        st.write("- Energy optimization")
        st.write("- Failure prediction")
        st.write("- Fraud and quality analytics")


def render_dashboard(reasoner: GPTReasoner) -> None:
    frame = st.session_state.frame
    profile = st.session_state.profile
    run = st.session_state.run

    st.divider()
    preview, quality = st.columns([1.4, 1])
    with preview:
        st.subheader("Dataset Preview")
        st.dataframe(frame.head(100), use_container_width=True)
    with quality:
        st.subheader("Dataset Intelligence")
        if profile:
            st.metric("Rows", profile["rows"])
            st.metric("Columns", profile["columns"])
            st.metric("Missing value rate", f"{profile['missing_value_rate']:.2%}")
            st.write("Target:", profile.get("target_variable") or "Not identified")
            st.write("Numeric:", ", ".join(profile["numeric_columns"][:8]) or "None")
            st.write("Categorical:", ", ".join(profile["categorical_columns"][:8]) or "None")
        else:
            st.write("Run analysis to generate dataset intelligence.")

    if not run or not profile:
        return

    tabs = st.tabs(["AI Summary", "Models", "Explainability", "Visuals", "Report", "Assistant"])

    with tabs[0]:
        st.subheader("AI Dataset Summary")
        st.write(st.session_state.ai.get("dataset", ""))
        st.subheader("Problem Type Detection")
        st.info(st.session_state.problem_type.replace("_", " ").title())
        st.write(st.session_state.ai.get("problem", ""))
        st.subheader("Generated Workflow")
        st.write(st.session_state.ai.get("workflow", ""))

    with tabs[1]:
        st.subheader("Recommended Models")
        st.write(st.session_state.ai.get("models", ""))
        st.dataframe(run.metrics, use_container_width=True)
        if not run.predictions.empty:
            st.subheader("Predictions")
            st.dataframe(run.predictions.head(100), use_container_width=True)
            st.download_button("Download predictions CSV", dataframe_download(run.predictions), "autods_gpt_predictions.csv", "text/csv")
        st.subheader("Performance Interpretation")
        st.write(st.session_state.ai.get("performance", ""))
        if run.confusion is not None:
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(run.confusion), use_container_width=True)

    with tabs[2]:
        importances = feature_importance(run)
        st.subheader("Feature Importance")
        if importances.empty:
            st.write("Feature importance is unavailable for the selected best model.")
        else:
            st.plotly_chart(px.bar(importances, x="importance", y="feature", orientation="h", title="Top Feature Drivers"), use_container_width=True)
            st.dataframe(importances, use_container_width=True)
        st.subheader("SHAP")
        shap_values, shap_note = shap_summary(run)
        st.write(shap_note)
        if not shap_values.empty:
            st.plotly_chart(px.bar(shap_values, x="mean_abs_shap", y="feature", orientation="h", title="Mean Absolute SHAP Impact"), use_container_width=True)
        st.subheader("LIME")
        st.write(lime_status())
        st.subheader("Root-Cause Analysis")
        st.write(st.session_state.ai.get("root_cause", ""))

    with tabs[3]:
        render_visuals(frame, profile, run)

    with tabs[4]:
        report_bytes = build_pdf_report(
            profile,
            st.session_state.problem_type,
            run.metrics,
            st.session_state.ai.get("performance", ""),
            st.session_state.ai.get("workflow", ""),
        )
        st.download_button("Download PDF report", report_bytes, "autods_gpt_report.pdf", "application/pdf")
        st.text_area(
            "Report summary",
            value=f"{st.session_state.ai.get('performance', '')}\n\n{st.session_state.ai.get('workflow', '')}",
            height=260,
        )

    with tabs[5]:
        st.subheader("Conversational AI Assistant")
        question = st.text_input("Ask about model failure, feature importance, accuracy, anomalies, forecasts, or business insight")
        if st.button("Ask AutoDS-GPT"):
            context = {
                "profile": profile,
                "problem_type": st.session_state.problem_type,
                "best_model": run.best_model_name,
                "metrics": run.metrics.to_dict(orient="records"),
                "ai_insights": st.session_state.ai,
            }
            st.write(reasoner.chat(question, context))


def render_visuals(frame: pd.DataFrame, profile: dict, run) -> None:
    numeric = profile.get("numeric_columns", [])
    if numeric:
        st.subheader("Numerical Distributions")
        selected = st.selectbox("Column", numeric)
        st.plotly_chart(px.histogram(frame, x=selected, marginal="box"), use_container_width=True)
    if len(numeric) > 1:
        st.subheader("Correlation Heatmap")
        corr = frame[numeric].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto"), use_container_width=True)
    if run.problem_type == "time_series_forecasting" and profile.get("datetime_columns") and run.target:
        st.subheader("Forecast Signal")
        time_col = profile["datetime_columns"][0]
        chart_frame = frame[[time_col, run.target]].dropna().sort_values(time_col)
        st.plotly_chart(px.line(chart_frame, x=time_col, y=run.target), use_container_width=True)
    if run.predictions is not None and {"actual", "predicted"}.issubset(run.predictions.columns):
        st.subheader("Actual vs Predicted")
        actual_numeric = pd.to_numeric(run.predictions["actual"], errors="coerce")
        predicted_numeric = pd.to_numeric(run.predictions["predicted"], errors="coerce")
        numeric_pairs = pd.DataFrame(
            {
                "actual": actual_numeric,
                "predicted": predicted_numeric,
            }
        ).dropna()

        if len(numeric_pairs) >= 2:
            fig = px.scatter(
                numeric_pairs,
                x="actual",
                y="predicted",
                trendline="ols",
                title="Actual vs Predicted",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            comparison = run.predictions.copy()
            comparison["match"] = comparison["actual"].astype(str) == comparison["predicted"].astype(str)
            match_counts = comparison["match"].map({True: "Correct", False: "Incorrect"}).value_counts().reset_index()
            match_counts.columns = ["result", "count"]
            st.plotly_chart(
                px.bar(match_counts, x="result", y="count", title="Prediction Match Summary"),
                use_container_width=True,
            )
            st.dataframe(comparison.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
