import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response, generate_insights_prompt
from utils.data_processor import profile_data

def render():
    st.header("AI-Powered Insights")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # API Key Configuration
    with st.expander("API Configuration", expanded=True):
        provider = st.radio("Select AI Provider", ["Anthropic (Claude)", "Google (Gemini)"])
        api_key = st.text_input("Enter API Key", type="password")

        provider_code = 'anthropic' if "Anthropic" in provider else 'google'

    if st.button("Generate Comprehensive Insights"):
        if not api_key:
            st.error("Please enter an API key.")
        else:
            with st.spinner("Analyzing data and generating insights..."):
                # Prepare summary
                profile = profile_data(df)

                # Simplified summary for LLM to avoid token limits
                summary_dict = {
                    'rows': profile['rows'],
                    'columns': profile['columns'],
                    'missing_values': profile['missing_total'],
                    'numeric_columns': list(profile['numeric_stats'].keys()) if profile['numeric_stats'] else [],
                    'correlations': profile['correlation']
                }

                # Sample data
                sample_data = df.head(5).to_markdown()

                prompt = generate_insights_prompt(str(summary_dict), f"Sample Data:\n{sample_data}")

                response = get_ai_response(prompt, api_key, provider_code)

                if "Error" in response:
                    st.error(response)
                else:
                    st.markdown("### Executive Summary")
                    st.markdown(response)

                    # Store insights for report
                    st.session_state['last_insights'] = response
