import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processor import profile_data, calculate_quality_score, detect_anomalies, detect_drift

def render():
    st.header("Data Profiling")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset in the 'Data Ingestion' page first.")
        return

    df = st.session_state['data']

    if st.button("Generate Profile"):
        with st.spinner("Profiling data..."):
            profile = profile_data(df)
            quality_score, quality_details = calculate_quality_score(df)

            # --- Overview & Quality Score ---
            st.subheader("Overview & Quality")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Rows", profile['rows'])
            col2.metric("Columns", profile['columns'])
            col3.metric("Duplicates", profile['duplicates'])
            col4.metric("Missing Values", profile['missing_total'])

            # Quality Score Gauge
            col5.metric("Quality Score", f"{quality_score}/100", delta="High" if quality_score > 80 else "Low")

            with st.expander("Quality Score Details"):
                st.write(quality_details)

            # --- Tabs ---
            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs(["Statistics", "Missing & Correlations", "Anomaly Detection", "Data Drift"])

            with tab1:
                # Numerical Stats
                st.subheader("Numerical Statistics")
                if profile['numeric_stats']:
                    st.dataframe(pd.DataFrame(profile['numeric_stats']))
                else:
                    st.info("No numerical columns found.")

                # Column Details
                st.subheader("Column Distribution")
                selected_col = st.selectbox("Select Column to visualize", df.columns)

                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    fig = px.histogram(df, x=selected_col, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.bar(df[selected_col].value_counts().reset_index(), x=selected_col, y='count')
                    fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Missing Values
                st.subheader("Missing Values by Column")
                missing_df = pd.DataFrame(list(profile['missing_by_col'].items()), columns=['Column', 'Missing Count'])
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                if not missing_df.empty:
                    st.bar_chart(missing_df.set_index('Column'))
                else:
                    st.success("No missing values found.")

                # Correlation Matrix
                st.subheader("Correlation Matrix")
                if profile['correlation']:
                    corr_df = pd.DataFrame(profile['correlation'])
                    fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough numerical data for correlation.")

            with tab3:
                st.subheader("Automated Anomaly Detection")
                st.write("Uses Isolation Forest to detect anomalies in numerical data.")

                # We run this on demand inside the tab
                if st.button("Run Anomaly Detection", key="run_anomaly"):
                    with st.spinner("Detecting anomalies..."):
                        df_anom, n_anomalies = detect_anomalies(df)
                        if n_anomalies > 0:
                            st.warning(f"Detected {n_anomalies} anomalies ({round(n_anomalies/len(df)*100, 2)}%)")
                            st.write("Top Anomalies (sorted by anomaly score):")
                            # Show anomalies sorted by score
                            st.dataframe(df_anom[df_anom['is_anomaly']].sort_values('anomaly_score').head(20))

                            # Visualization
                            num_cols = df.select_dtypes(include=[np.number]).columns
                            if len(num_cols) >= 2:
                                fig = px.scatter(df_anom, x=num_cols[0], y=num_cols[1], color='is_anomaly',
                                                title=f"Anomalies in {num_cols[0]} vs {num_cols[1]}",
                                                color_discrete_map={False: 'blue', True: 'red'})
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.success("No significant anomalies detected.")

            with tab4:
                st.subheader("Data Drift Analysis")
                st.info("Compares the first half of the dataset to the second half to check for distribution shifts.")

                drift_report = detect_drift(df)

                if drift_report:
                    drifted_cols = [col for col, data in drift_report.items() if data['drift_detected']]
                    if drifted_cols:
                        st.error(f"Drift detected in columns: {', '.join(drifted_cols)}")
                    else:
                        st.success("No significant drift detected.")

                    # Show details
                    st.write("Drift Metrics:")
                    st.json(drift_report)
                else:
                    st.info("Dataset too small for drift detection (need > 50 rows).")
