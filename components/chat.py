"""
Enhanced Chat Component
Features: Smooth UX, guided prompts, conversation history, plain language interactions.
"""

import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response, generate_chat_prompt, get_available_provider, LLM_PROVIDERS


def render():
    """Render the Chat page with guided experience."""

    if st.session_state['data'] is None:
        st.markdown("""
        <div class="help-tip">
            <strong>💬 Chat needs data to work with</strong><br>
            Upload a dataset first, then come back here to ask questions about it in plain English.
            The AI will analyze your data and give you answers - no coding required!
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Data Ingestion", type="primary"):
            st.session_state['current_page'] = "Data Ingestion"
            st.rerun()
        return

    df = st.session_state['data']

    # Auto-detect available provider from secrets/environment
    provider_code, api_key, selected_model = get_available_provider()

    # Show provider status
    with st.sidebar:
        st.subheader("Chat Configuration")
        if provider_code:
            provider_name = LLM_PROVIDERS[provider_code]['name']
            st.success(f"**{provider_name}** connected")
            st.caption(f"Model: {selected_model}")
        else:
            st.error("No AI provider configured")

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Example prompts when chat is empty
    if not st.session_state.messages:
        st.markdown("""
        <div class="help-tip">
            <strong>💡 What can you ask?</strong><br>
            Ask anything about your data in plain English. Here are some ideas to get started:
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # Organized prompt categories
        categories = {
            "Understand Your Data": [
                "What are the main patterns in this dataset?",
                "Give me a summary of the key statistics",
                "Which columns are most important?",
            ],
            "Find Issues": [
                "Are there any outliers I should worry about?",
                "What's the best way to handle the missing values?",
                "Are there any data quality issues?",
            ],
            "Get Insights": [
                "What interesting correlations exist in this data?",
                "What would be a good prediction target?",
                "What story does this data tell?",
            ],
        }

        for category, prompts in categories.items():
            st.markdown(f"**{category}:**")
            cols = st.columns(len(prompts))
            for i, prompt in enumerate(prompts):
                with cols[i]:
                    if st.button(prompt, key=f"example_{category}_{i}", use_container_width=True):
                        st.session_state.pending_prompt = prompt
                        st.rerun()

    # Check if provider is available
    if not provider_code:
        st.error("No AI provider configured. Please add an API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY) to your environment variables or Streamlit secrets.")
        return

    # Handle pending prompt from example buttons
    pending_prompt = st.session_state.pop('pending_prompt', None)

    # Chat input
    if prompt := (pending_prompt or st.chat_input("Ask anything about your data...")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                # Build data context
                data_context = f"""
Dataset Overview:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {df.columns.tolist()}

Data Types:
{df.dtypes.to_string()}

Basic Statistics:
{df.describe().to_string() if not df.select_dtypes(include=['number']).empty else 'No numeric columns'}

Sample Data (first 5 rows):
{df.head().to_markdown()}

Missing Values:
{df.isnull().sum().to_string()}
"""

                # Enhanced system context for non-technical users
                system_context = (
                    "You are a friendly data assistant helping someone who may not have a data science background. "
                    "Always explain things in plain, simple language. Avoid jargon unless you explain it. "
                    "When giving recommendations, be specific about what to do and why. "
                    "If the user asks about something complex, break it down into simple steps. "
                    "Use analogies and examples when helpful. "
                    "Format your responses with clear headings and bullet points for readability."
                )

                full_prompt = f"{system_context}\n\n" + generate_chat_prompt(
                    data_context,
                    prompt,
                    st.session_state.messages[:-1]
                )

                response = get_ai_response(
                    full_prompt,
                    api_key,
                    provider_code,
                    model=selected_model,
                    max_tokens=4096
                )

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
