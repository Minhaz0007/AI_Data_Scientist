import streamlit as st
import pandas as pd
from utils.data_loader import load_data, load_sql
import os

def render():
    st.header("Data Ingestion")

    tab1, tab2 = st.tabs(["File Upload", "SQL Database"])

    with tab1:
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls', 'json'])

        if uploaded_file is not None:
            try:
                # Determine file type
                file_type = uploaded_file.name.split('.')[-1].lower()
                if 'xls' in file_type:
                    file_type = 'excel'

                # Load data
                if st.button("Load File"):
                    with st.spinner('Loading data...'):
                        df = load_data(uploaded_file, file_type)
                        st.session_state['data'] = df
                        st.session_state['file_meta'] = {
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'type': file_type
                        }
                        st.success(f"Successfully loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with tab2:
        st.subheader("Connect to SQL Database")
        conn_string = st.text_input("Connection String (e.g., sqlite:///data.db)")
        query = st.text_area("SQL Query", "SELECT * FROM my_table")

        if st.button("Load from SQL"):
            if not conn_string or not query:
                st.error("Please provide both connection string and query.")
            else:
                try:
                    with st.spinner("Executing query..."):
                        df = load_sql(conn_string, query)
                        st.session_state['data'] = df
                        st.session_state['file_meta'] = {
                            'name': 'SQL Query Result',
                            'size': 0,
                            'type': 'sql'
                        }
                        st.success("Successfully loaded data from SQL.")
                except Exception as e:
                    st.error(f"Error loading from SQL: {e}")

    # Preview
    if st.session_state['data'] is not None:
        st.subheader("Data Preview")
        st.write(f"**File:** {st.session_state['file_meta']['name']}")
        st.write(f"**Shape:** {st.session_state['data'].shape}")

        st.dataframe(st.session_state['data'].head(100))

        if st.checkbox("Show Data Info"):
            st.write("Column Types:")
            st.write(st.session_state['data'].dtypes)
