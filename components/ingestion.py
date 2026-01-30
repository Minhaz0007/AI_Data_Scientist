"""
Enhanced Data Ingestion Component
Supports file upload, URL loading, SQL databases, sample datasets, and auto-detection.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_data, load_sql, load_url, load_api, load_sample
import os
import requests
from io import StringIO, BytesIO

# Sample datasets for quick start
SAMPLE_DATASETS = {
    "Iris (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic (Classification)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Tips (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "Diamonds (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
    "Penguins (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    "Flights (Time Series)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
    "Car Crashes (Analysis)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv",
    "MPG (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
}


def auto_detect_delimiter(content):
    """Auto-detect CSV delimiter."""
    delimiters = [',', ';', '\t', '|']
    sample = content[:5000] if len(content) > 5000 else content

    max_count = 0
    best_delimiter = ','

    for delim in delimiters:
        count = sample.count(delim)
        if count > max_count:
            max_count = count
            best_delimiter = delim

    return best_delimiter


def auto_detect_encoding(content_bytes):
    """Auto-detect file encoding."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            content_bytes.decode(encoding)
            return encoding
        except:
            continue

    return 'utf-8'


def load_from_url(url):
    """Load data from URL with auto-detection."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()

        # Determine file type from URL or content type
        if url.endswith('.csv') or 'csv' in content_type:
            encoding = auto_detect_encoding(response.content)
            content = response.content.decode(encoding)
            delimiter = auto_detect_delimiter(content)
            df = pd.read_csv(StringIO(content), sep=delimiter)

        elif url.endswith('.json') or 'json' in content_type:
            df = pd.read_json(BytesIO(response.content))

        elif url.endswith('.xlsx') or url.endswith('.xls') or 'excel' in content_type:
            df = pd.read_excel(BytesIO(response.content))

        elif url.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(response.content))

        else:
            # Try CSV as default
            encoding = auto_detect_encoding(response.content)
            content = response.content.decode(encoding)
            delimiter = auto_detect_delimiter(content)
            df = pd.read_csv(StringIO(content), sep=delimiter)

        return df, None

    except requests.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


def auto_optimize_dtypes(df):
    """Automatically optimize data types to reduce memory usage."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':
            # Check if it's a date
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
                continue
            except:
                pass

            # Check if it should be categorical
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

        elif col_type == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')

        elif col_type == 'int64':
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')

    return df


def get_data_summary(df):
    """Generate a quick data summary."""
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_total': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include='number').columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_cols': len(df.select_dtypes(include='datetime').columns),
    }
    return summary


def render():
    st.header("Data Ingestion")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["File Upload", "URL", "API", "SQL Database", "Sample Datasets"])

    # Tab 1: File Upload
    with tab1:
        render_file_upload()

    with tab2:
        render_url_upload()

    with tab3:
        render_api_upload()

    with tab4:
        render_sql_connection()

    with tab5:
        render_sample_datasets()

    # Data preview and info
    if st.session_state.get('data') is not None:
        render_data_preview()


def render_file_upload():
    st.info("Upload CSV, Excel (.xlsx, .xls, .xlsm), JSON, or Parquet files. No file size limits.")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset (drag and drop supported)",
            type=['csv', 'xlsx', 'xls', 'xlsm', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )

    with col2:
        optimize_types = st.checkbox("Auto-optimize types", value=True, help="Automatically optimize data types for memory efficiency")

    if uploaded_file is not None:
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                encoding = st.selectbox("Encoding", ['auto', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252'])
            with col2:
                delimiter = st.selectbox("Delimiter (CSV)", ['auto', ',', ';', '\\t', '|'])

        if st.button("Load File", type="primary"):
            try:
                with st.spinner('Loading and analyzing data...'):
                    file_type = uploaded_file.name.split('.')[-1].lower()

                    if file_type == 'csv':
                        content = uploaded_file.read()

                        # Auto-detect encoding
                        if encoding == 'auto':
                            encoding = auto_detect_encoding(content)

                        content_str = content.decode(encoding)

                        # Auto-detect delimiter
                        if delimiter == 'auto':
                            delimiter = auto_detect_delimiter(content_str)
                        elif delimiter == '\\t':
                            delimiter = '\t'

                        df = pd.read_csv(StringIO(content_str), sep=delimiter)

                    elif 'xls' in file_type:
                        df = pd.read_excel(uploaded_file)

                    elif file_type == 'json':
                        df = pd.read_json(uploaded_file)

                    elif file_type == 'parquet':
                        df = pd.read_parquet(uploaded_file)
                    else:
                        df = load_data(uploaded_file, file_type)

                    if optimize_types:
                        df = auto_optimize_dtypes(df)

                    st.session_state['data'] = df
                    st.session_state['file_meta'] = {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'type': file_type,
                        'source': 'file_upload'
                    }
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


def render_url_upload():
    st.info("Load data directly from a URL (CSV, JSON, Excel, or Parquet).")

    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com/data.csv",
        help="Direct link to a data file"
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        optimize_types = st.checkbox("Auto-optimize types", value=True, key="url_optimize")

    if url and st.button("Load from URL", type="primary"):
        with st.spinner("Fetching data from URL..."):
            df, error = load_from_url(url)

            if error:
                st.error(error)
            else:
                if optimize_types:
                    df = auto_optimize_dtypes(df)

                st.session_state['data'] = df
                st.session_state['file_meta'] = {
                    'name': url.split('/')[-1] or "URL Dataset",
                    'size': 0,
                    'type': "url"
                }
                st.success("Successfully loaded data from URL.")
                st.rerun()


def render_api_upload():
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
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading from API: {e}")


def render_sql_connection():
    st.subheader("Connect to SQL Database")
    conn_string = st.text_input("Connection String (e.g., sqlite:///data.db)")
    query = st.text_area("SQL Query", "SELECT * FROM my_table")

    if st.button("Load from SQL"):
        if not conn_string:
            st.error("Please provide connection string")
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
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading from SQL: {e}")


def render_sample_datasets():
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
                st.rerun()
        except Exception as e:
            st.error(f"Error loading sample: {e}")


def render_data_preview():
    st.subheader("Data Preview")

    if 'file_meta' in st.session_state and st.session_state['file_meta']:
        st.write(f"**File:** {st.session_state['file_meta']['name']}")

    df = st.session_state['data']
    st.write(f"**Shape:** {df.shape}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Drop Rows with Any Missing"):
            original_len = len(df)
            st.session_state['data'] = df.dropna()
            removed = original_len - len(st.session_state['data'])
            st.success(f"Removed {removed} rows with missing values")
            st.rerun()

    with col2:
        if st.button("Reset Data"):
            st.session_state['data'] = None
            st.session_state['file_meta'] = None
            st.success("Data reset")
            st.rerun()

    st.dataframe(df.head())

    summary = get_data_summary(df)
    st.json(summary)
