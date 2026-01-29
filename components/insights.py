import streamlit as st
import pandas as pd
import os
from utils.llm_helper import get_ai_response, generate_insights_prompt, LLM_PROVIDERS
from utils.data_processor import profile_data

def render():
    st.header("AI-Powered Insights")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # API Key Configuration
    with st.expander("API Configuration", expanded=True):
        provider_names = {k: v['name'] for k, v in LLM_PROVIDERS.items()}
        provider_display = st.radio("Select AI Provider", list(provider_names.values()))

        # Get provider code from display name
        provider_code = [k for k, v in provider_names.items() if v == provider_display][0]

        api_key = st.text_input("Enter API Key (Optional)", type="password", help="Leave blank if set in environment variables.")

        # Model selection
        available_models = LLM_PROVIDERS[provider_code]['models']
        default_model = LLM_PROVIDERS[provider_code]['default_model']
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=available_models.index(default_model)
        )

        # Token limit
        max_tokens = st.slider("Max Response Tokens", 1024, 8192, 4096)

    if st.button("Generate Comprehensive Insights"):
        has_env_key = False
        if provider_code == 'google' and os.environ.get('GOOGLE_API_KEY'): has_env_key = True
        elif provider_code == 'anthropic' and os.environ.get('ANTHROPIC_API_KEY'): has_env_key = True
        elif provider_code == 'openai' and os.environ.get('OPENAI_API_KEY'): has_env_key = True

        if not api_key and not has_env_key:
            st.error("Please enter an API key.")
        else:
            with st.spinner("Analyzing data and generating insights..."):
                # Prepare summary
                profile = profile_data(df)

                # Comprehensive summary for LLM
                summary_dict = {
                    'rows': profile['rows'],
                    'columns': profile['columns'],
                    'column_names': list(df.columns),
                    'data_types': profile['dtypes'],
                    'missing_values_total': profile['missing_total'],
                    'missing_by_column': {k: v for k, v in profile['missing_by_col'].items() if v > 0},
                    'duplicate_rows': profile['duplicates'],
                    'numeric_columns': list(profile['numeric_stats'].keys()) if profile['numeric_stats'] else [],
                    'numeric_statistics': profile['numeric_stats'],
                    'top_correlations': _get_top_correlations(profile['correlation']) if profile['correlation'] else None
                }

                # Sample data
                sample_data = df.head(10).to_markdown()

                prompt = generate_insights_prompt(str(summary_dict), f"Sample Data:\n{sample_data}")

                response = get_ai_response(
                    prompt,
                    api_key,
                    provider_code,
                    model=selected_model,
                    max_tokens=max_tokens
                )

                if "Error" in response:
                    st.error(response)
                else:
                    st.markdown("### Executive Summary")
                    st.markdown(response)

                    # Store insights for report
                    st.session_state['last_insights'] = response

def _get_top_correlations(corr_dict, top_n=5):
    """Extract top correlations from correlation dictionary."""
    if not corr_dict:
        return None

    correlations = []
    keys = list(corr_dict.keys())
    for i, col1 in enumerate(keys):
        for col2 in keys[i+1:]:
            corr_value = corr_dict[col1].get(col2, 0)
            if corr_value != 1.0:  # Skip self-correlations
                correlations.append({
                    'columns': f"{col1} <-> {col2}",
                    'correlation': round(corr_value, 3)
                })

    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return correlations[:top_n]
