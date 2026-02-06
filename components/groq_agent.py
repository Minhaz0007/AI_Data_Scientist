"""
Groq Chat Agent - Appears on every tab as a contextual AI assistant.
Provides suggestions on file load and can answer questions about data
in the context of the current tab.
"""

import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response, get_available_provider, LLM_PROVIDERS

# Tab-specific context and suggestion prompts
TAB_CONTEXT = {
    "Data Ingestion": {
        "role": "data ingestion specialist",
        "focus": "file formats, data sources, loading strategies, data quality first impressions",
        "suggestions_prompt": (
            "The user just loaded a dataset. As a data ingestion specialist, provide 5-7 SHORT, actionable suggestions. "
            "Cover: data format observations, encoding/delimiter issues, column naming conventions, "
            "immediate data quality red flags, recommended next steps for profiling. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Data Profiling": {
        "role": "data profiling expert",
        "focus": "statistical summaries, distributions, data types, missing values, correlations",
        "suggestions_prompt": (
            "The user is profiling this dataset. Provide 5-7 SHORT, actionable profiling suggestions. "
            "Cover: which columns to examine first, expected distributions, correlation hypotheses, "
            "data type mismatches to check, missing value patterns to investigate. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Data Cleaning": {
        "role": "data cleaning expert",
        "focus": "missing values, duplicates, outliers, data type corrections, standardization",
        "suggestions_prompt": (
            "The user wants to clean this dataset. Provide 5-7 SHORT, actionable cleaning suggestions. "
            "Cover: missing value strategy per column, duplicate detection, outlier handling, "
            "data type fixes, string standardization, column renaming. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Transformation": {
        "role": "data transformation specialist",
        "focus": "filtering, aggregation, pivoting, merging, calculated columns",
        "suggestions_prompt": (
            "The user wants to transform this dataset. Provide 5-7 SHORT, actionable transformation suggestions. "
            "Cover: useful filters, aggregation ideas, pivot possibilities, "
            "calculated columns to create, data reshaping opportunities. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Visualization": {
        "role": "data visualization expert",
        "focus": "chart types, visual encodings, storytelling with data, dashboard design",
        "suggestions_prompt": (
            "The user wants to visualize this dataset. Provide 5-7 SHORT, actionable visualization suggestions. "
            "Cover: best chart types for the columns, color encoding ideas, "
            "comparison charts, distribution plots, relationship visualizations. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Dashboard": {
        "role": "dashboard design expert",
        "focus": "KPI selection, layout design, interactive filters, real-time metrics",
        "suggestions_prompt": (
            "The user wants to build a dashboard. Provide 5-7 SHORT, actionable dashboard suggestions. "
            "Cover: key KPIs to track, chart combinations, filter recommendations, "
            "layout ideas, metric cards to pin. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Reporting": {
        "role": "data reporting specialist",
        "focus": "report structure, key findings, executive summaries, export formats",
        "suggestions_prompt": (
            "The user wants to generate a report. Provide 5-7 SHORT, actionable reporting suggestions. "
            "Cover: report sections to include, key findings to highlight, "
            "visualization selection for reports, executive summary points, export format choice. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Statistical Analysis": {
        "role": "statistician",
        "focus": "hypothesis testing, statistical tests, confidence intervals, p-values",
        "suggestions_prompt": (
            "The user wants statistical analysis. Provide 5-7 SHORT, actionable analysis suggestions. "
            "Cover: appropriate statistical tests, hypothesis ideas, "
            "normality checks needed, correlation analysis, group comparison tests. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Feature Engineering": {
        "role": "feature engineering expert",
        "focus": "feature creation, polynomial features, interactions, encoding, selection",
        "suggestions_prompt": (
            "The user wants to engineer features. Provide 5-7 SHORT, actionable suggestions. "
            "Cover: polynomial features to try, interaction terms, datetime extractions, "
            "encoding strategies for categoricals, feature selection methods. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Predictive Modeling": {
        "role": "machine learning engineer",
        "focus": "model selection, training, evaluation, hyperparameter tuning",
        "suggestions_prompt": (
            "The user wants to build predictive models. Provide 5-7 SHORT, actionable suggestions. "
            "Cover: recommended algorithms, target variable selection, train/test split strategy, "
            "evaluation metrics, potential overfitting concerns. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Time Series": {
        "role": "time series analyst",
        "focus": "trend analysis, seasonality, forecasting, ARIMA, decomposition",
        "suggestions_prompt": (
            "The user wants to do time series analysis. Provide 5-7 SHORT, actionable suggestions. "
            "Cover: datetime column identification, trend/seasonality detection, "
            "stationarity tests, forecasting model selection, lag features. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Advanced Analysis": {
        "role": "advanced analytics specialist",
        "focus": "PCA, t-SNE, anomaly detection, clustering, text analysis",
        "suggestions_prompt": (
            "The user wants advanced analysis. Provide 5-7 SHORT, actionable suggestions. "
            "Cover: dimensionality reduction candidates, anomaly detection approach, "
            "clustering feasibility, text analysis if applicable, recommended techniques. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Workflow Automation": {
        "role": "data pipeline architect",
        "focus": "automated pipelines, step sequencing, reproducibility, scheduling",
        "suggestions_prompt": (
            "The user wants to automate workflows. Provide 5-7 SHORT, actionable suggestions. "
            "Cover: pipeline steps to automate, recommended sequence, "
            "error handling considerations, reproducibility tips, scheduling ideas. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "AI Insights": {
        "role": "AI analyst",
        "focus": "comprehensive AI-powered analysis, pattern discovery, recommendations",
        "suggestions_prompt": (
            "The user wants AI insights. Provide 5-7 SHORT, actionable suggestions. "
            "Cover: analysis angles to explore, hidden patterns to look for, "
            "business questions the data can answer, anomalies to investigate. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
    "Chat": {
        "role": "data assistant",
        "focus": "natural language Q&A about data, exploration guidance",
        "suggestions_prompt": (
            "The user is chatting about their data. Provide 5-7 SHORT, interesting questions they could ask. "
            "Cover: data exploration questions, statistical queries, "
            "comparison questions, trend questions, actionable insight questions. "
            "Keep each suggestion to 1-2 sentences. Use bullet points."
        ),
    },
}


def _get_data_context(df, max_rows=5):
    """Build a compact data context string for the LLM."""
    numeric_stats = ""
    if not df.select_dtypes(include="number").empty:
        numeric_stats = df.describe().to_string()

    missing = df.isnull().sum()
    missing_info = missing[missing > 0].to_string() if missing.any() else "None"

    context = (
        f"Dataset: {len(df)} rows x {len(df.columns)} columns\n"
        f"Columns: {df.columns.tolist()}\n"
        f"Data Types:\n{df.dtypes.to_string()}\n\n"
        f"Statistics:\n{numeric_stats}\n\n"
        f"Sample ({max_rows} rows):\n{df.head(max_rows).to_markdown()}\n\n"
        f"Missing Values:\n{missing_info}\n"
        f"Duplicate Rows: {df.duplicated().sum()}"
    )
    return context


def _get_groq_provider():
    """Get Groq provider specifically, falling back to any available provider."""
    # Try Groq first
    import os
    groq_key = None
    try:
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            groq_key = st.secrets['GROQ_API_KEY']
    except Exception:
        pass
    if not groq_key:
        groq_key = os.environ.get('GROQ_API_KEY')

    if groq_key:
        return 'groq', groq_key, LLM_PROVIDERS['groq']['default_model']

    # Fall back to any available provider
    return get_available_provider()


def generate_suggestions(page_name):
    """Generate contextual AI suggestions for the current tab."""
    df = st.session_state.get('data')
    if df is None:
        return None

    provider_code, api_key, model = _get_groq_provider()
    if not provider_code:
        return None

    tab_info = TAB_CONTEXT.get(page_name, TAB_CONTEXT["Chat"])
    data_ctx = _get_data_context(df)

    prompt = (
        f"You are a {tab_info['role']}. Focus area: {tab_info['focus']}.\n\n"
        f"Here is the dataset context:\n{data_ctx}\n\n"
        f"{tab_info['suggestions_prompt']}"
    )

    try:
        response = get_ai_response(prompt, api_key, provider_code, model=model, max_tokens=1024)
        return response
    except Exception:
        return None


def render_agent(page_name):
    """Render the Groq chat agent panel for the current tab."""
    # Initialize per-tab chat history
    chat_key = f"groq_agent_messages_{page_name}"
    suggestions_key = f"groq_suggestions_{page_name}"
    suggestions_generated_key = f"groq_suggestions_generated_{page_name}"

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.markdown("---")
    st.markdown("### Groq AI Assistant")

    provider_code, api_key, model = _get_groq_provider()
    if provider_code:
        provider_name = LLM_PROVIDERS[provider_code]['name']
        st.caption(f"Powered by **{provider_name}** ({model})")
    else:
        st.warning(
            "No AI provider configured. Add a GROQ_API_KEY (or any supported API key) "
            "to your environment or Streamlit secrets to enable the AI assistant."
        )
        return

    df = st.session_state.get('data')

    # ── Auto-suggestions on data load ──
    if df is not None:
        # Track whether suggestions have been generated for this data + tab combo
        data_fingerprint = f"{len(df)}_{len(df.columns)}_{hash(tuple(df.columns))}"
        gen_key = f"{suggestions_generated_key}_{data_fingerprint}"

        if not st.session_state.get(gen_key, False):
            with st.spinner("Generating suggestions..."):
                suggestions = generate_suggestions(page_name)
                if suggestions:
                    st.session_state[suggestions_key] = suggestions
                    st.session_state[gen_key] = True

        # Show suggestions
        if st.session_state.get(suggestions_key):
            with st.expander("AI Suggestions", expanded=True):
                st.markdown(st.session_state[suggestions_key])
                if st.button("Refresh Suggestions", key=f"refresh_sugg_{page_name}"):
                    with st.spinner("Refreshing..."):
                        suggestions = generate_suggestions(page_name)
                        if suggestions:
                            st.session_state[suggestions_key] = suggestions
                            st.rerun()
    else:
        st.info("Upload a dataset to get AI-powered suggestions for this tab.")

    # ── Chat interface ──
    with st.expander("Chat", expanded=len(st.session_state[chat_key]) > 0):
        # Display chat history
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Quick prompts when chat is empty
        if not st.session_state[chat_key] and df is not None:
            tab_info = TAB_CONTEXT.get(page_name, TAB_CONTEXT["Chat"])
            st.caption(f"Ask me anything about {tab_info['focus']}")

        if df is not None:
            user_input = st.text_input(
                "Ask the AI assistant...",
                key=f"groq_input_{page_name}",
                placeholder=f"Ask about {TAB_CONTEXT.get(page_name, TAB_CONTEXT['Chat'])['focus']}..."
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                send = st.button("Send", key=f"groq_send_{page_name}", type="primary")
            with col2:
                if st.button("Clear Chat", key=f"groq_clear_{page_name}"):
                    st.session_state[chat_key] = []
                    st.rerun()

            if send and user_input:
                st.session_state[chat_key].append({"role": "user", "content": user_input})

                tab_info = TAB_CONTEXT.get(page_name, TAB_CONTEXT["Chat"])
                data_ctx = _get_data_context(df)

                # Build conversation history
                history = ""
                if st.session_state[chat_key]:
                    history = "\n\nConversation so far:\n"
                    for msg in st.session_state[chat_key][-6:]:
                        history += f"{msg['role'].upper()}: {msg['content']}\n"

                prompt = (
                    f"You are a {tab_info['role']}. The user is on the '{page_name}' tab. "
                    f"Focus area: {tab_info['focus']}.\n\n"
                    f"Dataset context:\n{data_ctx}\n"
                    f"{history}\n"
                    f"User question: {user_input}\n\n"
                    f"Provide a helpful, concise answer. If code is needed, use Python/pandas snippets."
                )

                with st.spinner("Thinking..."):
                    response = get_ai_response(prompt, api_key, provider_code, model=model, max_tokens=2048)

                st.session_state[chat_key].append({"role": "assistant", "content": response})
                st.rerun()
