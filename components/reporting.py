import streamlit as st
import pandas as pd
from utils.report_generator import generate_html_report, generate_pdf_report, generate_docx_report

def render():
    st.header("Report Generation")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # Report title
    report_title = st.text_input("Report Title", "Data Analysis Report")

    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )

    with col2:
        # Export as Excel
        try:
            import io
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")

    with col3:
        # Export as JSON
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="processed_data.json",
            mime="application/json"
        )

    st.markdown("---")

    # AI Insights status
    st.subheader("Analysis Report")

    insights = st.session_state.get('last_insights', None)

    if insights:
        st.success("AI insights available and will be included in the report.")
        with st.expander("Preview Insights"):
            st.markdown(insights)
    else:
        st.warning("No AI insights generated yet. Go to 'AI Insights' to generate them for a complete report.")

    st.markdown("---")

    # Report configuration
    st.subheader("Generate Report")

    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        report_template = st.selectbox("Report Template", ["Standard", "Executive Summary", "Technical Deep Dive"])
    with col_conf2:
         report_format = st.radio(
            "Select Format",
            ["HTML (Recommended)", "PDF", "Word (DOCX)"],
            horizontal=True
        )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("✨ Auto-Generate Report", type="primary"):
            st.info(f"Generating {report_template} report...")
            report_title = f"{report_title} - {report_template}"
            with st.spinner(f"Generating {report_format.split()[0]} report..."):
                try:
                    if "HTML" in report_format:
                        report_data = generate_html_report(df, insights, report_title)
                        file_name = "analysis_report.html"
                        mime_type = "text/html"
                        st.session_state['generated_report'] = {
                            'data': report_data,
                            'name': file_name,
                            'mime': mime_type,
                            'format': 'HTML'
                        }
                        st.success("HTML report generated successfully!")

                    elif "PDF" in report_format:
                        report_data = generate_pdf_report(df, insights, report_title)
                        file_name = "analysis_report.pdf"
                        mime_type = "application/pdf"
                        st.session_state['generated_report'] = {
                            'data': report_data,
                            'name': file_name,
                            'mime': mime_type,
                            'format': 'PDF'
                        }
                        st.success("PDF report generated successfully!")

                    elif "Word" in report_format:
                        report_data = generate_docx_report(df, insights, report_title)
                        file_name = "analysis_report.docx"
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        st.session_state['generated_report'] = {
                            'data': report_data,
                            'name': file_name,
                            'mime': mime_type,
                            'format': 'DOCX'
                        }
                        st.success("Word document generated successfully!")

                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Download button for generated report
    if 'generated_report' in st.session_state:
        report = st.session_state['generated_report']
        with col2:
            st.download_button(
                label=f"Download {report['format']} Report",
                data=report['data'],
                file_name=report['name'],
                mime=report['mime'],
                type="primary"
            )

    # Report contents preview (HTML only)
    if 'generated_report' in st.session_state and st.session_state['generated_report']['format'] == 'HTML':
        with st.expander("Preview HTML Report"):
            st.components.v1.html(st.session_state['generated_report']['data'], height=600, scrolling=True)
