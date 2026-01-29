import streamlit as st
import pandas as pd
import os
from utils.llm_helper import get_ai_response, generate_chat_prompt, LLM_PROVIDERS

def render():
    st.header("Chat with your Data")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # API Key Configuration in sidebar
    with st.sidebar:
        st.subheader("Chat Configuration")
        provider_names = {k: v['name'] for k, v in LLM_PROVIDERS.items()}
        provider_display = st.radio("Provider", list(provider_names.values()), key="chat_provider", index=list(provider_names.keys()).index('google') if 'google' in provider_names else 0)
        provider_code = [k for k, v in provider_names.items() if v == provider_display][0]

        api_key = st.text_input("API Key (Optional if set in Env)", type="password", key="chat_api_key", help="Leave blank to use environment variable.")

        available_models = LLM_PROVIDERS[provider_code]['models']
        selected_model = st.selectbox(
            "Model",
            available_models,
            index=0,
            key="chat_model"
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Example prompts
    if not st.session_state.messages:
        st.info("Try asking questions like:")
        example_prompts = [
            "What are the main trends in this data?",
            "Show me the correlation between different columns",
            "What are the outliers in this dataset?",
            "Summarize the key statistics",
            "How can I clean this data?"
        ]
        cols = st.columns(2)
        for i, prompt in enumerate(example_prompts):
            with cols[i % 2]:
                if st.button(prompt, key=f"example_{i}"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

    # Handle pending prompt from example buttons
    pending_prompt = st.session_state.pop('pending_prompt', None)

    # Chat input
    if prompt := (pending_prompt or st.chat_input("Ask a question about your data")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Check for API key in input or environment (handled by get_ai_response, but we check here to show UI error if both missing)
            has_env_key = False
            if provider_code == 'google' and os.environ.get('GOOGLE_API_KEY'): has_env_key = True
            elif provider_code == 'anthropic' and os.environ.get('ANTHROPIC_API_KEY'): has_env_key = True
            elif provider_code == 'openai' and os.environ.get('OPENAI_API_KEY'): has_env_key = True

            if not api_key and not has_env_key:
                st.error("Please enter an API key in the sidebar or set it in the environment.")
                st.session_state.messages.append({"role": "assistant", "content": "Error: Please enter an API key."})
            else:
                with st.spinner("Thinking..."):
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

                    full_prompt = generate_chat_prompt(
                        data_context,
                        prompt,
                        st.session_state.messages[:-1]  # Exclude current message
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
