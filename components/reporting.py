import streamlit as st
import pandas as pd
from utils.report_generator import generate_html_report

def render():
    st.header("Report Generation")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    st.subheader("Export Options")

    # Export Cleaned Data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Dataset (CSV)",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Export Report
    st.subheader("Analysis Report")

    insights = st.session_state.get('last_insights', None)

    if insights:
        st.info("Insights from the 'AI Insights' section will be included.")
    else:
        st.warning("No AI insights generated yet. Go to 'AI Insights' to generate them.")

    if st.button("Generate HTML Report"):
        html_report = generate_html_report(df, insights)

        st.download_button(
            label="Download HTML Report",
            data=html_report,
            file_name="analysis_report.html",
            mime="text/html"
        )
