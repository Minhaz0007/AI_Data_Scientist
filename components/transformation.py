import streamlit as st
import pandas as pd
from utils.data_processor import filter_data, group_and_aggregate

def render():
    st.header("Data Transformation")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    tab1, tab2 = st.tabs(["Filter", "Group & Aggregate"])

    with tab1:
        st.subheader("Filter Data")
        col_to_filter = st.selectbox("Select Column to Filter", df.columns)
        condition = st.selectbox("Condition", ['equals', 'greater_than', 'less_than', 'contains'])
        value = st.text_input("Value")

        if st.button("Apply Filter"):
            try:
                filtered_df = filter_data(df, col_to_filter, condition, value)
                st.write(f"Filtered Data ({len(filtered_df)} rows):")
                st.dataframe(filtered_df.head())

                if st.button("Save Filtered Data as New Dataset"):
                    st.session_state['data'] = filtered_df
                    st.success("Dataset updated.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error filtering: {e}")

    with tab2:
        st.subheader("Group by and Aggregate")
        group_col = st.selectbox("Group By", df.columns)
        agg_col = st.selectbox("Aggregate Column", df.select_dtypes(include='number').columns)
        method = st.selectbox("Aggregation Method", ['mean', 'sum', 'count', 'min', 'max'])

        if st.button("Aggregate"):
            try:
                result = group_and_aggregate(df, group_col, agg_col, method)
                st.write("Result:")
                st.dataframe(result)
            except Exception as e:
                st.error(f"Error aggregating: {e}")
