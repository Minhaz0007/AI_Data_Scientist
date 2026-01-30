import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response, generate_chat_prompt, get_available_provider, LLM_PROVIDERS

def render():
    st.header("Chat with your Data")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # Auto-detect available provider from secrets/environment
    provider_code, api_key, selected_model = get_available_provider()

    # Show provider status in sidebar
    with st.sidebar:
        st.subheader("Chat Configuration")
        if provider_code:
            provider_name = LLM_PROVIDERS[provider_code]['name']
            st.info(f"Using **{provider_name}**\nModel: **{selected_model}**")
        else:
            st.error("No AI provider configured")

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

    # Check if provider is available before allowing chat
    if not provider_code:
        st.error("No AI provider configured. Please add an API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY) to your Streamlit secrets or environment variables.")
        return

    # Handle pending prompt from example buttons
    pending_prompt = st.session_state.pop('pending_prompt', None)

    # Chat input
    if prompt := (pending_prompt or st.chat_input("Ask a question about your data")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
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
