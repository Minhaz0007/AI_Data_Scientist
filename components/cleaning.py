import streamlit as st
import pandas as pd
from utils.data_processor import remove_duplicates, impute_missing, normalize_column_names, get_missing_summary

def render():
    st.header("Data Cleaning")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    st.subheader("Current Dataset Info")
    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))

    st.markdown("---")

    # Duplicates
    st.subheader("Duplicate Removal")
    duplicates_count = df.duplicated().sum()
    st.write(f"Total Duplicates: {duplicates_count}")
    if duplicates_count > 0:
        if st.button("Remove Duplicates"):
            st.session_state['data'] = remove_duplicates(df)
            st.success(f"Removed {duplicates_count} duplicates.")
            st.rerun()

    st.markdown("---")

    # Missing Values
    st.subheader("Handle Missing Values")
    missing_summary = get_missing_summary(df)

    if missing_summary:
        st.write("Columns with missing values:")
        st.dataframe(pd.DataFrame(list(missing_summary.items()), columns=['Column', 'Missing Count']))

        col_to_clean = st.selectbox("Select Column", list(missing_summary.keys()))
        strategy = st.selectbox("Imputation Strategy", ['mean', 'median', 'mode', 'constant', 'drop'])

        fill_value = None
        if strategy == 'constant':
            fill_value = st.text_input("Enter Constant Value")

        if st.button("Apply Imputation"):
            try:
                st.session_state['data'] = impute_missing(
                    st.session_state['data'],
                    [col_to_clean],
                    strategy,
                    fill_value
                )
                st.success(f"Imputed column {col_to_clean} using {strategy}.")
                st.rerun()
            except Exception as e:
                st.error(f"Error imputing: {e}")
    else:
        st.success("No missing values found.")

    st.markdown("---")

    # Column Normalization
    st.subheader("Standardize Column Names")
    if st.button("Normalize to Snake Case"):
        st.session_state['data'] = normalize_column_names(df)
        st.success("Column names normalized.")
        st.rerun()

    # Show Data
    st.subheader("Cleaned Data Preview")
    st.dataframe(st.session_state['data'].head())
