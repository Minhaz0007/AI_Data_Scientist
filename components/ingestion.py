"""
Enhanced Data Ingestion Component
Supports file upload, URL loading, SQL databases, sample datasets, and auto-detection.
Files are persisted in the database and survive app refreshes.
"""

import streamlit as st
import pandas as pd
import pyarrow as pa
from utils.data_loader import load_data, load_sql, load_url, load_api, load_sample
from utils.db import (
    save_uploaded_file, load_uploaded_files_list,
    load_uploaded_file_data, delete_uploaded_file
)
import os
import requests
import warnings
from io import StringIO
from io import BytesIO

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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
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


def _persist_data(df, file_name, file_type, file_size=0, source='file_upload'):
    """Save data to session state and persist to database."""
    st.session_state['data'] = df
    st.session_state['file_meta'] = {
        'name': file_name,
        'size': file_size,
        'type': file_type,
        'source': source
    }
    # Persist to database
    save_uploaded_file(df, file_name, file_type, file_size=file_size, source=source)


def render():
    # Quick-start for new users (guided mode)
    if st.session_state.get('guided_mode', True) and st.session_state.get('data') is None:
        st.markdown("""
        <div class="help-tip">
            <strong>👋 Getting Started</strong><br>
            Pick how you want to bring your data in. The easiest way is to <strong>upload a file</strong> from your computer,
            or try a <strong>sample dataset</strong> to explore the app first!
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    # Show recent files section first
    render_recent_files()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 File Upload", "🔗 URL", "🌐 API", "🗄️ SQL Database", "📦 Sample Datasets"
    ])

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


def render_recent_files():
    """Show recent files stored in the database."""
    files = load_uploaded_files_list()
    if not files:
        return

    st.subheader("Recent Files")
    st.caption("Previously uploaded files are stored and available across sessions.")

    # Display as a table with actions
    for idx, f in enumerate(files):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        with col1:
            display_name = f['file_name']
            if len(display_name) > 35:
                display_name = display_name[:32] + "..."
            st.markdown(f"**{display_name}**")
        with col2:
            st.caption(f"{f['row_count']} rows x {f['col_count']} cols | {f['file_type']}")
        with col3:
            if st.button("Load", key=f"load_file_{f['id']}_{idx}", type="primary"):
                with st.spinner(f"Loading {f['file_name']}..."):
                    df, name, ftype = load_uploaded_file_data(f['id'])
                    if df is not None:
                        st.session_state['data'] = df
                        st.session_state['file_meta'] = {
                            'name': name,
                            'size': f.get('file_size', 0),
                            'type': ftype,
                            'source': f.get('source', 'database')
                        }
                        st.success(f"Loaded {name}")
                        st.rerun()
                    else:
                        st.error("Failed to load file from database.")
        with col4:
            if st.button("Delete", key=f"delete_file_{f['id']}_{idx}"):
                if delete_uploaded_file(f['id']):
                    # If the currently loaded file is the one being deleted, clear it
                    current_meta = st.session_state.get('file_meta')
                    if current_meta and current_meta.get('name') == f['file_name']:
                        st.session_state['data'] = None
                        st.session_state['file_meta'] = None
                    st.success(f"Deleted {f['file_name']}")
                    st.rerun()
                else:
                    st.error("Failed to delete file.")

    st.markdown("---")


def render_file_upload():
    st.markdown("**Drag and drop** your file below, or click **Browse files**. Supported: CSV, Excel, JSON, Parquet.")

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

                    _persist_data(df, uploaded_file.name, file_type, file_size=uploaded_file.size, source='file_upload')
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

                file_name = url.split('/')[-1] or "URL Dataset"
                _persist_data(df, file_name, 'url', source='url')
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
                    _persist_data(df, "API Data", 'api', source='api')
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
                    _persist_data(df, 'SQL Query Result', 'sql', source='sql')
                    st.success("Successfully loaded data from SQL.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading from SQL: {e}")


def render_sample_datasets():
    st.markdown("**Try the app instantly** with a pre-loaded dataset. Great for exploring features!")
    st.markdown("")

    # Display datasets as cards
    dataset_info = {
        "Iris (Classification)": {"desc": "Flower measurements - great for learning classification", "rows": "150", "icon": "🌸"},
        "Titanic (Classification)": {"desc": "Passenger survival data - predict who survived", "rows": "891", "icon": "🚢"},
        "Tips (Regression)": {"desc": "Restaurant tipping data - predict tip amounts", "rows": "244", "icon": "🍽️"},
        "Diamonds (Regression)": {"desc": "Diamond pricing data - predict diamond prices", "rows": "53,940", "icon": "💎"},
        "Penguins (Classification)": {"desc": "Penguin species data - classify penguin types", "rows": "344", "icon": "🐧"},
        "Flights (Time Series)": {"desc": "Monthly airline passengers - forecast trends", "rows": "144", "icon": "✈️"},
        "Car Crashes (Analysis)": {"desc": "US car crash statistics - analyze safety patterns", "rows": "51", "icon": "🚗"},
        "MPG (Regression)": {"desc": "Car fuel efficiency - predict miles per gallon", "rows": "398", "icon": "⛽"},
    }

    cols = st.columns(4)
    for i, (name, info) in enumerate(dataset_info.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="step-card" style="margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-size: 1.3rem;">{info['icon']}</div>
                <strong style="font-size: 0.85rem;">{name.split('(')[0].strip()}</strong>
                <p style="color: var(--text-secondary); font-size: 0.75rem; margin: 4px 0;">{info['desc']}</p>
                <span style="color: var(--text-muted); font-size: 0.7rem;">{info['rows']} rows</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load {name.split('(')[0].strip()}", key=f"sample_{name}", use_container_width=True):
                url = SAMPLE_DATASETS.get(name)
                if url:
                    try:
                        with st.spinner(f"Loading {name}..."):
                            df, error = load_from_url(url)
                            if df is not None:
                                _persist_data(df, f"{name.split('(')[0].strip()}.csv", 'csv', source='sample')
                                st.success(f"Loaded {name}!")
                                st.rerun()
                            else:
                                st.error(f"Error: {error}")
                    except Exception as e:
                        st.error(f"Error loading sample: {e}")


def render_data_preview():
    st.markdown("---")
    st.subheader("Data Preview")

    df = st.session_state['data']
    summary = get_data_summary(df)

    # KPI row
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Rows", f"{summary['rows']:,}")
    with kpi_cols[1]:
        st.metric("Columns", f"{summary['columns']}")
    with kpi_cols[2]:
        st.metric("Missing Values", f"{summary['missing_total']:,}")
    with kpi_cols[3]:
        st.metric("Duplicates", f"{summary['duplicates']:,}")
    with kpi_cols[4]:
        st.metric("Memory", f"{summary['memory_mb']:.1f} MB")

    # Column types summary
    type_cols = st.columns(3)
    with type_cols[0]:
        st.caption(f"**{summary['numeric_cols']}** numeric columns")
    with type_cols[1]:
        st.caption(f"**{summary['categorical_cols']}** text/category columns")
    with type_cols[2]:
        st.caption(f"**{summary['datetime_cols']}** date/time columns")

    st.markdown("")

    # Data table with controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        preview_rows = st.slider("Preview rows", 5, min(100, len(df)), 10, key="preview_rows")
    with col2:
        if st.button("Drop Missing Rows", help="Remove all rows that have any missing values"):
            original_len = len(df)
            st.session_state['data'] = df.dropna()
            removed = original_len - len(st.session_state['data'])
            st.success(f"Removed {removed} rows")
            st.rerun()
    with col3:
        if st.button("Clear Data", help="Remove the current dataset"):
            st.session_state['data'] = None
            st.session_state['file_meta'] = None
            st.rerun()

    try:
        st.dataframe(df.head(preview_rows), use_container_width=True)
    except pa.ArrowInvalid:
        st.dataframe(df.head(preview_rows).astype(str), use_container_width=True)
