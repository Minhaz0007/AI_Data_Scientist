import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response

def render():
    st.header("Chat with your Data")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # API Key Configuration (Reuse or ask again, for simplicity asking again or checking session)
    # Better to have global config but I'll stick to local for simplicity in this artifact
    api_key = st.sidebar.text_input("API Key for Chat", type="password")
    provider = st.sidebar.radio("Provider", ["anthropic", "google"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your data (e.g., 'What is the trend in sales?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key:
                st.error("Please enter an API key in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    # Context building
                    data_context = f"Dataset Columns: {df.columns.tolist()}\n"
                    data_context += f"First 5 rows:\n{df.head().to_markdown()}\n"
                    data_context += f"Data Types:\n{df.dtypes.to_string()}\n"

                    full_prompt = f"""
                    You are a data assistant. Here is the context of the dataset:
                    {data_context}

                    User Question: {prompt}

                    Answer the question based on the data provided. If you need to perform complex analysis, describe the steps.
                    """

                    response = get_ai_response(full_prompt, api_key, provider)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
