import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processor import profile_data

def render():
    st.header("Data Profiling")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset in the 'Data Ingestion' page first.")
        return

    df = st.session_state['data']

    if st.button("Generate Profile"):
        with st.spinner("Profiling data..."):
            profile = profile_data(df)

            # Overview
            st.subheader("Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", profile['rows'])
            col2.metric("Columns", profile['columns'])
            col3.metric("Duplicates", profile['duplicates'])
            col4.metric("Missing Values", profile['missing_total'])

            # Missing Values
            st.subheader("Missing Values by Column")
            missing_df = pd.DataFrame(list(profile['missing_by_col'].items()), columns=['Column', 'Missing Count'])
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.bar_chart(missing_df.set_index('Column'))
            else:
                st.success("No missing values found.")

            # Numerical Stats
            st.subheader("Numerical Statistics")
            if profile['numeric_stats']:
                st.dataframe(pd.DataFrame(profile['numeric_stats']))
            else:
                st.info("No numerical columns found.")

            # Correlation Matrix
            st.subheader("Correlation Matrix")
            if profile['correlation']:
                corr_df = pd.DataFrame(profile['correlation'])
                fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numerical data for correlation.")

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
