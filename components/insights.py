"""
Enhanced AI Insights Component
Features: One-click insights, friendly explanations, stored results, guided prompts.
"""

import streamlit as st
import pandas as pd
from utils.llm_helper import get_ai_response, generate_insights_prompt, get_available_provider, LLM_PROVIDERS
from utils.data_processor import profile_data


def render():
    if st.session_state['data'] is None:
        st.markdown("""
        <div class="help-tip">
            <strong>💡 AI Insights needs data</strong><br>
            Upload a dataset first, then come back here. Our AI will read through your entire dataset
            and write a comprehensive analysis with key findings and actionable recommendations.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Data Ingestion", type="primary"):
            st.session_state['current_page'] = "Data Ingestion"
            st.rerun()
        return

    df = st.session_state['data']

    # Auto-detect available provider
    provider_code, api_key, selected_model = get_available_provider()

    # Provider status
    if provider_code:
        provider_name = LLM_PROVIDERS[provider_code]['name']
        st.markdown(f"""
        <div class="status-indicator status-saved" style="margin-bottom: 1rem;">
            ✓ Connected to {provider_name} ({selected_model})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No AI provider configured. Please add an API key to your environment or Streamlit secrets.")
        return

    # Configuration
    with st.expander("Settings", expanded=False):
        max_tokens = st.slider("Max Response Length", 1024, 8192, 4096,
                               help="Higher values give more detailed analysis but take longer")

    # Show previously generated insights
    if st.session_state.get('last_insights'):
        st.markdown("### Previous Analysis")
        st.markdown(st.session_state['last_insights'])
        st.markdown("---")

    # Generate button
    col1, col2 = st.columns([2, 1])
    with col1:
        generate_btn = st.button(
            "Generate AI Analysis" if not st.session_state.get('last_insights') else "Regenerate Analysis",
            type="primary",
            use_container_width=True
        )
    with col2:
        if st.session_state.get('last_insights'):
            if st.button("Clear Analysis", use_container_width=True):
                st.session_state['last_insights'] = None
                st.rerun()

    if generate_btn:
        with st.spinner("AI is analyzing your data... This may take a moment."):
            profile = profile_data(df)

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

            sample_data = df.head(10).to_markdown()

            # Enhanced prompt for non-technical users
            enhanced_prompt = (
                "IMPORTANT: Write this analysis for someone who may NOT have a data science background. "
                "Use plain, simple language. Explain technical concepts when you must use them. "
                "Structure your response with clear sections:\n"
                "1. Quick Summary (2-3 sentences overview)\n"
                "2. Key Findings (5-7 bullet points of the most interesting things)\n"
                "3. Data Quality (any issues to fix, in plain language)\n"
                "4. Recommendations (specific next steps the user should take)\n"
                "5. Interesting Patterns (correlations, trends, or anomalies)\n\n"
                "Be friendly and encouraging. Avoid jargon.\n\n"
            )

            prompt = enhanced_prompt + generate_insights_prompt(str(summary_dict), f"Sample Data:\n{sample_data}")

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
                st.markdown("### AI Analysis Results")
                st.markdown(response)
                st.session_state['last_insights'] = response
                st.success("Analysis complete! Results have been saved and will be included in reports.")


def _get_top_correlations(corr_dict, top_n=5):
    """Extract top correlations from correlation dictionary."""
    if not corr_dict:
        return None

    correlations = []
    keys = list(corr_dict.keys())
    for i, col1 in enumerate(keys):
        for col2 in keys[i+1:]:
            corr_value = corr_dict[col1].get(col2, 0)
            if corr_value != 1.0:
                correlations.append({
                    'columns': f"{col1} <-> {col2}",
                    'correlation': round(corr_value, 3)
                })

    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return correlations[:top_n]
