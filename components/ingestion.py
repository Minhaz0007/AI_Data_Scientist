import streamlit as st
import pandas as pd
from utils.data_loader import load_data, load_sql, load_url, load_api, load_sample
import os

def render():
    st.header("Data Ingestion")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["File Upload", "URL", "API", "SQL Database", "Sample Datasets"])

    # Tab 1: File Upload
    with tab1:
        st.info("No file size limits. Upload CSV, Excel (.xlsx, .xls, .xlsm), or JSON files.")
        uploaded_file = st.file_uploader(
            "Upload your dataset (drag and drop supported)",
            type=['csv', 'xlsx', 'xls', 'xlsm', 'json'],
            help="Supported formats: CSV, Excel (xlsx, xls, xlsm), JSON"
        )

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

    # Tab 2: URL
    with tab2:
        st.subheader("Load from URL")
        url = st.text_input("Enter Dataset URL", placeholder="https://example.com/data.csv")
        file_type_url = st.selectbox("File Type", ['csv', 'json', 'excel'])

        if st.button("Load from URL"):
            if not url:
                st.error("Please enter a URL.")
            else:
                try:
                    with st.spinner("Fetching data..."):
                        df = load_url(url, file_type_url)
                        st.session_state['data'] = df
                        st.session_state['file_meta'] = {
                            'name': url.split('/')[-1] or "URL Dataset",
                            'size': 0,
                            'type': f"url_{file_type_url}"
                        }
                        st.success("Successfully loaded data from URL.")
                except Exception as e:
                    st.error(f"Error loading from URL: {e}")

    # Tab 3: API
    with tab3:
        st.subheader("Load from API")
        api_url = st.text_input("API Endpoint", placeholder="https://api.example.com/data")
        json_key = st.text_input("JSON Key (optional)", help="Key to extract data from nested JSON (e.g. 'results' or 'data.items')")

        if st.button("Load from API"):
            if not api_url:
                st.error("Please enter an API URL.")
            else:
                try:
                    with st.spinner("Fetching API data..."):
                        df = load_api(api_url, json_key=json_key if json_key else None)
                        st.session_state['data'] = df
                        st.session_state['file_meta'] = {
                            'name': "API Data",
                            'size': 0,
                            'type': "api"
                        }
                        st.success("Successfully loaded data from API.")
                except Exception as e:
                    st.error(f"Error loading from API: {e}")

    # Tab 4: SQL
    with tab4:
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

    # Tab 5: Sample Datasets
    with tab5:
        st.subheader("Sample Datasets")
        dataset = st.selectbox("Select Dataset", ["Titanic", "Iris", "Housing", "Wine Quality"])

        if st.button("Load Sample"):
            try:
                with st.spinner(f"Loading {dataset}..."):
                    # Map display name to key
                    key_map = {
                        "Titanic": "titanic",
                        "Iris": "iris",
                        "Housing": "housing",
                        "Wine Quality": "wine"
                    }
                    df = load_sample(key_map[dataset])
                    st.session_state['data'] = df
                    st.session_state['file_meta'] = {
                        'name': f"{dataset} Sample",
                        'size': 0,
                        'type': 'sample'
                    }
                    st.success(f"Successfully loaded {dataset} dataset.")
            except Exception as e:
                st.error(f"Error loading sample: {e}")

    # Preview
    if st.session_state['data'] is not None:
        st.subheader("Data Preview")
        st.write(f"**File:** {st.session_state['file_meta']['name']}")
        st.write(f"**Shape:** {st.session_state['data'].shape}")

        st.dataframe(st.session_state['data'].head(100))

        if st.checkbox("Show Data Info"):
            st.write("Column Types:")
            st.write(st.session_state['data'].dtypes)
