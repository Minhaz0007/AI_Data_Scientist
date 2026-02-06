"""
Enhanced AI Assistant - Appears on every tab as a contextual AI navigation helper.
Provides suggestions on file load, helps users understand what to do,
and can answer questions about data in the context of the current tab.
"""

import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response, get_available_provider, LLM_PROVIDERS, _clean_api_key

# Tab-specific context and suggestion prompts
TAB_CONTEXT = {
    "Data Ingestion": {
        "role": "friendly data ingestion guide",
        "focus": "file formats, data sources, loading strategies, data quality first impressions",
        "suggestions_prompt": (
            "The user just loaded a dataset. As a friendly guide (assume no technical background), provide 5-7 SHORT, actionable suggestions. "
            "Explain in plain language: what the data looks like, if there are any obvious issues, "
            "and what they should do next. Avoid jargon. Use bullet points."
        ),
        "help_text": "I can help you understand your data after you upload it. I'll tell you what I see and suggest next steps.",
    },
    "Data Profiling": {
        "role": "friendly data exploration guide",
        "focus": "statistical summaries, distributions, data types, missing values, correlations",
        "suggestions_prompt": (
            "The user is exploring this dataset. Provide 5-7 SHORT suggestions in plain language. "
            "Tell them: which columns look interesting, what patterns you see, "
            "whether there are quality issues, and what to check next. No jargon."
        ),
        "help_text": "I'll help you understand what your data looks like - the patterns, quality issues, and interesting things to explore.",
    },
    "Data Cleaning": {
        "role": "friendly data cleaning guide",
        "focus": "missing values, duplicates, outliers, data type corrections, standardization",
        "suggestions_prompt": (
            "The user wants to clean this dataset. Provide 5-7 SHORT, practical suggestions in plain language. "
            "Tell them: what needs fixing, which approach to use for each issue, "
            "and explain WHY in simple terms. No jargon."
        ),
        "help_text": "I'll suggest how to fix issues in your data - like filling gaps, removing duplicates, and fixing errors.",
    },
    "Transformation": {
        "role": "friendly data transformation guide",
        "focus": "filtering, aggregation, pivoting, merging, calculated columns",
        "suggestions_prompt": (
            "The user wants to transform this dataset. Provide 5-7 SHORT suggestions in plain language. "
            "Suggest: useful ways to filter or reshape the data, "
            "new columns they could create, and how to organize data better. Keep it simple."
        ),
        "help_text": "I'll help you reshape and organize your data - filtering, combining, and creating new columns.",
    },
    "Visualization": {
        "role": "friendly visualization guide",
        "focus": "chart types, visual encodings, storytelling with data, dashboard design",
        "suggestions_prompt": (
            "The user wants to visualize this dataset. Provide 5-7 SHORT suggestions in plain language. "
            "Recommend: the best chart types for their specific columns, "
            "what comparisons would be interesting, and how to tell a story with the data."
        ),
        "help_text": "I'll suggest the best ways to visualize your data - which charts to use and what stories they can tell.",
    },
    "Dashboard": {
        "role": "friendly dashboard guide",
        "focus": "KPI selection, layout design, interactive filters, real-time metrics",
        "suggestions_prompt": (
            "The user is building a dashboard. Provide 5-7 SHORT suggestions in plain language. "
            "Recommend: key numbers to track, best chart combinations, "
            "and how to organize the dashboard for maximum clarity."
        ),
        "help_text": "I'll help you build an effective dashboard - choosing the right metrics and layout.",
    },
    "Reporting": {
        "role": "friendly report generation guide",
        "focus": "report structure, key findings, executive summaries, export formats",
        "suggestions_prompt": (
            "The user wants to generate a report. Provide 5-7 SHORT suggestions in plain language. "
            "Recommend: what to include in the report, how to structure findings, "
            "and which export format to choose."
        ),
        "help_text": "I'll help you create a professional report - what to include and how to present your findings.",
    },
    "Statistical Analysis": {
        "role": "friendly statistics guide",
        "focus": "hypothesis testing, statistical tests, confidence intervals, p-values",
        "suggestions_prompt": (
            "The user wants statistical analysis. Provide 5-7 SHORT suggestions in VERY plain language. "
            "Explain what tests would be useful and WHY in simple terms. "
            "Avoid statistical jargon - explain concepts like you would to a friend."
        ),
        "help_text": "I'll explain statistical tests in plain language and help you understand what they mean for your data.",
    },
    "Feature Engineering": {
        "role": "friendly feature creation guide",
        "focus": "feature creation, polynomial features, interactions, encoding, selection",
        "suggestions_prompt": (
            "The user wants to create new features. Provide 5-7 SHORT suggestions in plain language. "
            "Explain: what new columns would help predictions, "
            "how to extract useful info from dates/text, and which features matter most."
        ),
        "help_text": "I'll help you create new data points that can improve your predictions and analysis.",
    },
    "Predictive Modeling": {
        "role": "friendly machine learning guide",
        "focus": "model selection, training, evaluation, hyperparameter tuning",
        "suggestions_prompt": (
            "The user wants to build prediction models. Provide 5-7 SHORT suggestions in plain language. "
            "Recommend: what to predict, which algorithm to try first, "
            "how to know if the model is good, and common pitfalls to avoid. No jargon."
        ),
        "help_text": "I'll guide you through building AI models - what to predict, which approach to use, and how to evaluate results.",
    },
    "Time Series": {
        "role": "friendly time series guide",
        "focus": "trend analysis, seasonality, forecasting, ARIMA, decomposition",
        "suggestions_prompt": (
            "The user wants to analyze time-based data. Provide 5-7 SHORT suggestions in plain language. "
            "Help them: identify which column has dates, spot trends and patterns, "
            "and choose the right forecasting approach."
        ),
        "help_text": "I'll help you find trends over time and forecast what might happen next.",
    },
    "Advanced Analysis": {
        "role": "friendly advanced analytics guide",
        "focus": "PCA, t-SNE, anomaly detection, clustering, text analysis",
        "suggestions_prompt": (
            "The user wants advanced analysis. Provide 5-7 SHORT suggestions in plain language. "
            "Explain: which advanced techniques would help, what they do in simple terms, "
            "and what insights they can reveal."
        ),
        "help_text": "I'll explain advanced analysis techniques in simple terms and help you find hidden patterns.",
    },
    "Workflow Automation": {
        "role": "friendly automation guide",
        "focus": "automated pipelines, step sequencing, reproducibility, scheduling",
        "suggestions_prompt": (
            "The user wants to automate their workflow. Provide 5-7 SHORT suggestions in plain language. "
            "Recommend: what steps to automate, the best order, "
            "and how to set up a reliable pipeline."
        ),
        "help_text": "I'll help you set up an automated pipeline so you can process data with one click.",
    },
    "AI Insights": {
        "role": "friendly AI analysis guide",
        "focus": "comprehensive AI-powered analysis, pattern discovery, recommendations",
        "suggestions_prompt": (
            "The user wants AI insights. Provide 5-7 SHORT suggestions in plain language. "
            "Tell them: what angles to explore, what hidden patterns to look for, "
            "and what business questions the data can answer."
        ),
        "help_text": "I'll help you get the most from AI analysis - what to look for and how to interpret results.",
    },
    "Chat": {
        "role": "friendly data conversation guide",
        "focus": "natural language Q&A about data, exploration guidance",
        "suggestions_prompt": (
            "The user is chatting about their data. Provide 5-7 SHORT, interesting questions they could ask. "
            "Make them conversational and practical - things a business person would want to know."
        ),
        "help_text": "Just type a question in plain English and I'll analyze your data to find the answer!",
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
        return 'groq', _clean_api_key(groq_key), LLM_PROVIDERS['groq']['default_model']

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
        f"IMPORTANT: The user may not have any data science background. "
        f"Explain everything in simple, friendly language. No jargon.\n\n"
        f"Here is the dataset context:\n{data_ctx}\n\n"
        f"{tab_info['suggestions_prompt']}"
    )

    try:
        response = get_ai_response(prompt, api_key, provider_code, model=model, max_tokens=1024)
        if response and response.startswith("Error generating response:"):
            return None
        return response
    except Exception:
        return None


def render_agent(page_name):
    """Render the AI assistant panel for the current tab."""
    chat_key = f"groq_agent_messages_{page_name}"
    suggestions_key = f"groq_suggestions_{page_name}"
    suggestions_generated_key = f"groq_suggestions_generated_{page_name}"

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.markdown("---")

    # Modern header
    tab_info = TAB_CONTEXT.get(page_name, TAB_CONTEXT["Chat"])

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### AI Assistant")
        st.caption(tab_info.get('help_text', 'Ask me anything about your data.'))
    with col2:
        provider_code, api_key, model = _get_groq_provider()
        if provider_code:
            provider_name = LLM_PROVIDERS[provider_code]['name']
            st.markdown(f"""
            <div class="status-indicator status-saved" style="margin-top: 8px;">
                ✓ {provider_name}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-error" style="margin-top: 8px;">
                ✗ No AI configured
            </div>
            """, unsafe_allow_html=True)

    if not provider_code:
        st.markdown("""
        <div class="help-tip">
            <strong>🔧 Setup Required</strong><br>
            Add an API key (GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY)
            to your environment or Streamlit secrets to enable the AI assistant.
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.get('data')

    # ── Auto-suggestions on data load ──
    if df is not None:
        data_fingerprint = f"{len(df)}_{len(df.columns)}_{hash(tuple(df.columns))}"
        gen_key = f"{suggestions_generated_key}_{data_fingerprint}"

        if not st.session_state.get(gen_key, False):
            with st.spinner("Generating smart suggestions..."):
                suggestions = generate_suggestions(page_name)
                if suggestions:
                    st.session_state[suggestions_key] = suggestions
                    st.session_state[gen_key] = True

        # Show suggestions
        if st.session_state.get(suggestions_key):
            with st.expander("AI Suggestions for This Page", expanded=True):
                st.markdown(st.session_state[suggestions_key])
                if st.button("Refresh Suggestions", key=f"refresh_sugg_{page_name}"):
                    with st.spinner("Refreshing..."):
                        suggestions = generate_suggestions(page_name)
                        if suggestions:
                            st.session_state[suggestions_key] = suggestions
                            st.rerun()
                        else:
                            st.warning(
                                "Could not refresh suggestions. The API rate limit may "
                                "have been reached. Please wait a few minutes and try again."
                            )
    else:
        st.markdown("""
        <div class="help-tip">
            <strong>📂 Upload data to unlock AI suggestions</strong><br>
            Once you load a dataset, I'll automatically analyze it and give you personalized suggestions for this page.
        </div>
        """, unsafe_allow_html=True)

    # ── Chat interface ──
    with st.expander("Ask Me Anything", expanded=len(st.session_state[chat_key]) > 0):
        # Display chat history
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Quick prompts when chat is empty
        if not st.session_state[chat_key] and df is not None:
            st.caption(f"Try asking about: {tab_info['focus']}")

        if df is not None:
            user_input = st.text_input(
                "Ask the AI assistant...",
                key=f"groq_input_{page_name}",
                placeholder=f"e.g., What should I do with this data on this page?"
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

                data_ctx = _get_data_context(df)

                # Build conversation history
                history = ""
                if st.session_state[chat_key]:
                    history = "\n\nConversation so far:\n"
                    for msg in st.session_state[chat_key][-6:]:
                        history += f"{msg['role'].upper()}: {msg['content']}\n"

                prompt = (
                    f"You are a {tab_info['role']}. The user is on the '{page_name}' page. "
                    f"Focus area: {tab_info['focus']}.\n\n"
                    f"IMPORTANT: The user may not have data science experience. "
                    f"Always explain in plain, simple language. No jargon without explanation.\n\n"
                    f"Dataset context:\n{data_ctx}\n"
                    f"{history}\n"
                    f"User question: {user_input}\n\n"
                    f"Provide a helpful, friendly, concise answer. If code is needed, use Python/pandas snippets and explain what they do."
                )

                with st.spinner("Thinking..."):
                    response = get_ai_response(prompt, api_key, provider_code, model=model, max_tokens=2048)

                if response and "Rate limit reached" in response:
                    st.session_state[chat_key].append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    st.session_state[chat_key].append({"role": "assistant", "content": response})
                st.rerun()
