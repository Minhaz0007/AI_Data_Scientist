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
    st.session_state['dark_mode'] = False  # Default to light mode
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Data Ingestion"
if 'dashboard_charts' not in st.session_state:
    st.session_state['dashboard_charts'] = []

# AMOLED/XDR Optimized Dark Theme CSS
def apply_theme():
    if st.session_state.get('dark_mode', True):
        st.markdown("""
        <style>
        /* AMOLED Dark Theme - Pure black for OLED/XDR displays */
        :root {
            --bg-primary: #000000;
            --bg-secondary: #0a0a0a;
            --bg-tertiary: #262730;
            --bg-hover: #1f1f1f;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --accent: #3b82f6;
            --accent-light: #60a5fa;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --border: #4a4a4a;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stApp, [data-testid="stAppViewContainer"] {
            animation: fadeIn 0.4s ease-out;
        }

        /* Card Styling */
        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Main backgrounds */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
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

        /* Text */
        .stMarkdown, p, span, label, li {
            color: var(--text-primary) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }

        /* Navigation headers */
        .nav-header {
            color: var(--text-muted) !important;
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            padding: 0.75rem 0 0.25rem 0.5rem !important;
            margin: 0 !important;
        }

        /* Active nav item */
        .nav-item-active {
            background-color: var(--accent-glow) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--accent-light) !important;
            padding: 0.5rem 0.75rem !important;
            margin: 0.1rem 0 !important;
            border-radius: 0 6px 6px 0 !important;
            font-weight: 500 !important;
        }

        /* Sidebar nav buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border: none !important;
            text-align: left !important;
            padding: 0.5rem 0.75rem !important;
            font-weight: 400 !important;
            justify-content: flex-start !important;
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: var(--bg-hover) !important;
            color: var(--text-primary) !important;
            transform: translateX(4px) !important;
        }

        /* Main content buttons */
        [data-testid="stAppViewContainer"] .stButton > button {
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stAppViewContainer"] .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }

        /* Primary buttons */
        [data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
        [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, var(--accent), #2563eb) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
        }

        [data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 15px var(--accent-glow) !important;
        }

        /* Secondary buttons */
        [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
        }

        [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):hover {
            background-color: var(--bg-hover) !important;
            border-color: var(--accent) !important;
        }

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stMultiSelect > div > div > div {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            transition: border-color 0.2s;
        }

        .stSelectbox > div > div > div[data-baseweb="select"] > div {
             color: var(--text-primary) !important;
        }

        /* Dropdown options */
        ul[data-baseweb="menu"] li {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px var(--accent-glow) !important;
        }

        /* DataFrame styling */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--accent-light) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid var(--border) !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent-light) !important;
            border-bottom: 2px solid var(--accent) !important;
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

        .divider {
            height: 1px;
            background: var(--border);
            margin: 1rem 0;
        }

        .page-desc {
            color: var(--text-secondary) !important;
            font-size: 0.95rem;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme
        st.markdown("""
        <style>
        :root {
            --accent: #2563eb;
            --accent-light: #3b82f6;
            --accent-glow: rgba(37, 99, 235, 0.15);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stApp {
            animation: fadeIn 0.4s ease-out;
        }

        .nav-header {
            color: #666 !important;
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            padding: 0.75rem 0 0.25rem 0.5rem !important;
        }

        .nav-item-active {
            background-color: var(--accent-glow) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--accent) !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 0 6px 6px 0 !important;
            font-weight: 500 !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background-color: transparent !important;
            color: #555 !important;
            border: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #f0f0f0 !important;
            color: #333 !important;
        }

        [data-testid="stAppViewContainer"] .stButton > button:hover {
            transform: translateY(-2px);
        }

        [data-testid="stMetricValue"] {
            color: var(--accent) !important;
        }

        .divider {
            height: 1px;
            background: #e0e0e0;
            margin: 1rem 0;
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
        st.markdown("## 🔬 Data Engine")
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
from components import modeling, timeseries, feature_engineering, advanced_analysis, dashboard
from utils.db import init_db, save_project, load_projects, load_project_details

# Navigation structure
PAGES = {
    "Data Pipeline": ["Data Ingestion", "Data Profiling", "Data Cleaning", "Transformation"],
    "Data Science": ["Feature Engineering", "Predictive Modeling", "Time Series", "Advanced Analysis"],
    "Analytics & AI": ["Analysis", "Visualization", "Dashboard", "AI Insights", "Chat"],
    "Output": ["Reporting"]
}

PAGE_ICONS = {
    "Data Ingestion": "📤", "Data Profiling": "📊", "Data Cleaning": "🧹", "Transformation": "🔄",
    "Feature Engineering": "⚙️", "Predictive Modeling": "🎯", "Time Series": "📈", "Advanced Analysis": "🧠",
    "Analysis": "📉", "Visualization": "📊", "Dashboard": "🖥️", "AI Insights": "💡", "Chat": "💬",
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
    "Dashboard": "Pin your favorite visualizations to a custom dashboard",
    "AI Insights": "Get comprehensive AI-powered analysis and recommendations",
    "Chat": "Ask natural language questions about your data",
    "Reporting": "Generate and export professional reports in multiple formats"
}

# Sidebar
with st.sidebar:
    # Header with theme toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 🔬 Data Engine")
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
elif page == "Dashboard":
    dashboard.render()
elif page == "AI Insights":
    insights.render()
elif page == "Chat":
    chat.render()
elif page == "Reporting":
    reporting.render()
