import streamlit as st
import yaml
import os

# Load configuration
def load_config():
    with open('config/settings.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Page config
st.set_page_config(
    page_title=config['app_name'],
    page_icon="🔬",
    layout=config['layout'],
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'file_meta' not in st.session_state:
    st.session_state['file_meta'] = None
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True  # Default to dark mode
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Data Ingestion"

# AMOLED/XDR Optimized Dark Theme CSS
def apply_theme():
    if st.session_state.get('dark_mode', True):
        st.markdown("""
        <style>
        /* AMOLED Dark Theme - Pure black for OLED/XDR displays */
        :root {
            --bg-primary: #000000;
            --bg-secondary: #0d0d0d;
            --bg-tertiary: #1a1a1a;
            --bg-card: #0d0d0d;
            --text-primary: #ffffff;
            --text-secondary: #888888;
            --text-muted: #555555;
            --accent: #00d4aa;
            --accent-hover: #00f5c4;
            --accent-glow: rgba(0, 212, 170, 0.3);
            --border: #222222;
            --success: #00c853;
            --warning: #ffd600;
            --error: #ff5252;
            --info: #2196f3;
        }

        /* Main app background */
        .stApp {
            background-color: var(--bg-primary) !important;
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary) !important;
        }

        [data-testid="stHeader"] {
            background-color: var(--bg-primary) !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }

        section[data-testid="stSidebar"] > div {
            background-color: var(--bg-secondary) !important;
        }

        /* All text */
        .stMarkdown, p, span, label, li, div {
            color: var(--text-primary) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }

        /* Sidebar navigation */
        .nav-header {
            color: var(--accent) !important;
            font-size: 0.7rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.15em !important;
            padding: 1rem 0 0.5rem 0 !important;
            margin: 0 !important;
        }

        .nav-item {
            display: block;
            padding: 0.6rem 0.8rem;
            margin: 0.15rem 0;
            border-radius: 8px;
            color: var(--text-secondary) !important;
            text-decoration: none;
            transition: all 0.2s ease;
            font-size: 0.9rem;
        }

        .nav-item:hover {
            background-color: var(--bg-tertiary);
            color: var(--text-primary) !important;
        }

        .nav-item-active {
            background: linear-gradient(135deg, var(--accent-glow), transparent) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--accent) !important;
            font-weight: 600;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent), #00b894) !important;
            color: #000 !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 20px var(--accent-glow) !important;
        }

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            background-color: var(--bg-tertiary) !important;
            border-color: var(--border) !important;
        }

        [data-baseweb="select"] {
            background-color: var(--bg-tertiary) !important;
        }

        [data-baseweb="popover"] {
            background-color: var(--bg-tertiary) !important;
        }

        /* Multiselect */
        .stMultiSelect > div > div {
            background-color: var(--bg-tertiary) !important;
            border-color: var(--border) !important;
        }

        /* DataFrames */
        .stDataFrame {
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        [data-testid="stDataFrame"] > div {
            background-color: var(--bg-secondary) !important;
        }

        /* Tables */
        .stDataFrame table {
            background-color: var(--bg-secondary) !important;
        }

        .stDataFrame th {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        .stDataFrame td {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-color: var(--border) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            gap: 0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 0.5rem 1rem !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--bg-tertiary) !important;
            color: var(--accent) !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            background-color: var(--bg-tertiary) !important;
            border-radius: 0 8px 8px 8px !important;
            padding: 1rem !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: var(--bg-tertiary) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }

        .streamlit-expanderContent {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--accent) !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }

        [data-testid="stMetricDelta"] svg {
            fill: var(--accent) !important;
        }

        /* Alerts */
        .stAlert {
            background-color: var(--bg-tertiary) !important;
            border-radius: 8px !important;
        }

        [data-testid="stAlertContentInfo"] {
            background-color: rgba(33, 150, 243, 0.1) !important;
            border-left: 4px solid var(--info) !important;
        }

        [data-testid="stAlertContentSuccess"] {
            background-color: rgba(0, 200, 83, 0.1) !important;
            border-left: 4px solid var(--success) !important;
        }

        [data-testid="stAlertContentWarning"] {
            background-color: rgba(255, 214, 0, 0.1) !important;
            border-left: 4px solid var(--warning) !important;
        }

        [data-testid="stAlertContentError"] {
            background-color: rgba(255, 82, 82, 0.1) !important;
            border-left: 4px solid var(--error) !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: var(--bg-secondary) !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: var(--bg-tertiary) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: var(--accent) !important;
        }

        /* Radio buttons */
        .stRadio > div {
            gap: 0.25rem !important;
        }

        .stRadio label {
            background-color: transparent !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 6px !important;
            transition: all 0.2s ease !important;
        }

        .stRadio label:hover {
            background-color: var(--bg-tertiary) !important;
        }

        /* Checkbox */
        .stCheckbox {
            color: var(--text-primary) !important;
        }

        /* Slider */
        .stSlider [data-baseweb="slider"] {
            background-color: var(--bg-tertiary) !important;
        }

        .stSlider [data-testid="stTickBar"] {
            background: linear-gradient(90deg, var(--accent), #00b894) !important;
        }

        /* Progress */
        .stProgress > div > div {
            background-color: var(--accent) !important;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: var(--accent) !important;
        }

        /* Plotly charts */
        .js-plotly-plot .plotly .modebar {
            background-color: var(--bg-secondary) !important;
        }

        .js-plotly-plot .plotly .modebar-btn path {
            fill: var(--text-secondary) !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Custom divider */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border), transparent);
            margin: 1rem 0;
        }

        /* Data status card */
        .data-status {
            background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-card));
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--border);
        }

        /* Page description */
        .page-desc {
            color: var(--text-secondary) !important;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme
        st.markdown("""
        <style>
        :root {
            --accent: #FF4B4B;
            --accent-hover: #ff6b6b;
            --accent-glow: rgba(255, 75, 75, 0.2);
        }

        .nav-header {
            color: #666 !important;
            font-size: 0.7rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.15em !important;
            padding: 1rem 0 0.5rem 0 !important;
        }

        .nav-item {
            display: block;
            padding: 0.6rem 0.8rem;
            margin: 0.15rem 0;
            border-radius: 8px;
            color: #666 !important;
            font-size: 0.9rem;
        }

        .nav-item-active {
            background: linear-gradient(135deg, var(--accent-glow), transparent) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--accent) !important;
            font-weight: 600;
        }

        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--accent) !important;
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
            margin: 1rem 0;
        }

        .data-status {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)

# Apply theme
apply_theme()

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return True
    if st.session_state['authenticated']:
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔬 AI Data Scientist")
        st.markdown("Enter password to continue")
        entered_password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter password...")
        if st.button("Login", type="primary", use_container_width=True):
            if entered_password == password:
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

# Import components
from components import ingestion, profiling, cleaning, transformation, analysis, visualization, insights, chat, reporting
from components import modeling, timeseries, feature_engineering, advanced_analysis
from utils.db import init_db, save_project, load_projects, load_project_details

# Navigation structure
PAGES = {
    "Data Pipeline": ["Data Ingestion", "Data Profiling", "Data Cleaning", "Transformation"],
    "Data Science": ["Feature Engineering", "Predictive Modeling", "Time Series", "Advanced Analysis"],
    "Analytics & AI": ["Analysis", "Visualization", "AI Insights", "Chat"],
    "Output": ["Reporting"]
}

PAGE_ICONS = {
    "Data Ingestion": "📤", "Data Profiling": "📊", "Data Cleaning": "🧹", "Transformation": "🔄",
    "Feature Engineering": "⚙️", "Predictive Modeling": "🎯", "Time Series": "📈", "Advanced Analysis": "🧠",
    "Analysis": "📉", "Visualization": "📊", "AI Insights": "💡", "Chat": "💬",
    "Reporting": "📄"
}

PAGE_DESC = {
    "Data Ingestion": "Upload your data from CSV, Excel, JSON, or connect to a SQL database",
    "Data Profiling": "Explore statistics, distributions, correlations, and data quality",
    "Data Cleaning": "Handle missing values, remove duplicates, and standardize columns",
    "Transformation": "Filter, aggregate, pivot, merge datasets, and create calculated columns",
    "Feature Engineering": "Create polynomial features, interactions, datetime features, and select important features",
    "Predictive Modeling": "Train and evaluate machine learning models for regression and classification",
    "Time Series": "Analyze trends, seasonality, and forecast future values",
    "Advanced Analysis": "Perform PCA, t-SNE visualization, anomaly detection, and text analysis",
    "Analysis": "Run statistical tests, perform clustering, and detect outliers",
    "Visualization": "Create custom interactive charts and get AI-suggested visualizations",
    "AI Insights": "Get comprehensive AI-powered analysis and recommendations",
    "Chat": "Ask natural language questions about your data",
    "Reporting": "Generate and export professional reports in multiple formats"
}

# Sidebar
with st.sidebar:
    # Header with theme toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 🔬 AI Data Scientist")
    with col2:
        theme_btn = "☀️" if st.session_state.get('dark_mode') else "🌙"
        if st.button(theme_btn, help="Toggle theme"):
            st.session_state['dark_mode'] = not st.session_state.get('dark_mode', True)
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Data status
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        file_name = st.session_state['file_meta']['name']
        if len(file_name) > 25:
            file_name = file_name[:22] + "..."
        st.markdown(f"**📁 {file_name}**")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
    else:
        st.info("📂 Upload data to get started")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Navigation
    for section, pages in PAGES.items():
        st.markdown(f'<p class="nav-header">{section}</p>', unsafe_allow_html=True)
        for page in pages:
            icon = PAGE_ICONS.get(page, "")
            is_active = st.session_state.get('current_page') == page

            if is_active:
                st.markdown(f'<div class="nav-item nav-item-active">{icon} {page}</div>', unsafe_allow_html=True)
            else:
                if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                    st.session_state['current_page'] = page
                    st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Compact settings
    with st.expander("⚙️ Settings", expanded=False):
        st.caption("**Project Management**")
        if os.environ.get("DATABASE_URL"):
            proj_name = st.text_input("Name", label_visibility="collapsed", placeholder="Project name...")
            if st.button("💾 Save Project", use_container_width=True):
                if st.session_state['data'] is not None and proj_name:
                    summary = {"rows": len(st.session_state['data']), "cols": len(st.session_state['data'].columns)}
                    if save_project(proj_name, summary, st.session_state.get('last_insights', '')):
                        st.success("Saved!")
        else:
            st.caption("Set DATABASE_URL to enable")

    # Version
    st.caption(f"v{config['version']} • {'🌙 Dark' if st.session_state.get('dark_mode') else '☀️ Light'}")

# Main content
page = st.session_state.get('current_page', "Data Ingestion")
icon = PAGE_ICONS.get(page, "")
desc = PAGE_DESC.get(page, "")

st.markdown(f"# {icon} {page}")
st.markdown(f'<p class="page-desc">{desc}</p>', unsafe_allow_html=True)

# Render page
if page == "Data Ingestion":
    ingestion.render()
elif page == "Data Profiling":
    profiling.render()
elif page == "Data Cleaning":
    cleaning.render()
elif page == "Transformation":
    transformation.render()
elif page == "Feature Engineering":
    feature_engineering.render()
elif page == "Predictive Modeling":
    modeling.render()
elif page == "Time Series":
    timeseries.render()
elif page == "Advanced Analysis":
    advanced_analysis.render()
elif page == "Analysis":
    analysis.render()
elif page == "Visualization":
    visualization.render()
elif page == "AI Insights":
    insights.render()
elif page == "Chat":
    chat.render()
elif page == "Reporting":
    reporting.render()
