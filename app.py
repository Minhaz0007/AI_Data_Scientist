import streamlit as st
import yaml
import os
import json
from datetime import datetime

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

# ─── Initialize Session State ────────────────────────────────────────
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'file_meta' not in st.session_state:
    st.session_state['file_meta'] = None
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False  # Light mode only
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Data Ingestion"
if 'dashboard_charts' not in st.session_state:
    st.session_state['dashboard_charts'] = []
if 'onboarding_complete' not in st.session_state:
    st.session_state['onboarding_complete'] = False
if 'show_help_panel' not in st.session_state:
    st.session_state['show_help_panel'] = False
if 'auto_save_status' not in st.session_state:
    st.session_state['auto_save_status'] = None
if 'last_saved_at' not in st.session_state:
    st.session_state['last_saved_at'] = None
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = []
if 'user_preferences' not in st.session_state:
    st.session_state['user_preferences'] = {}
if 'guided_mode' not in st.session_state:
    st.session_state['guided_mode'] = True
if 'notifications' not in st.session_state:
    st.session_state['notifications'] = []
if 'completed_steps' not in st.session_state:
    st.session_state['completed_steps'] = set()


# ─── Modern Theme CSS ────────────────────────────────────────────────
def apply_theme():
    # ─── Light-Only Theme with Overlay Fix ───
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ═══════════════════════════════════════════════════════════
       FIX: Hide Streamlit sidebar collapse button icon text leak
       The Material Icons ligature text (keyboard_double_arrow_left)
       renders as raw text when the font fails to load.
       We hide the text color and replace it with an SVG background.
       ═══════════════════════════════════════════════════════════ */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        font-size: 0 !important;
        overflow: hidden !important;
    }

    /* Target the icon element which contains the leaking text */
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="collapsedControl"] button span,
    [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] button [data-testid="stIconMaterial"],
    [data-testid="stExpander"] [data-testid="stIconMaterial"] {
        color: transparent !important; /* Hide the text "ub", "arrow...", etc */
        font-size: 1.25rem !important;
        overflow: hidden !important;
        display: inline-flex !important;
        width: 24px !important;
        height: 24px !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
    }

    /* Inject SVG icon replacements */
    [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"]::after,
    [data-testid="collapsedControl"] button [data-testid="stIconMaterial"]::after,
    [data-testid="stExpander"] [data-testid="stIconMaterial"]::after,
    [data-testid="stSidebarCollapseButton"] button span::after,
    [data-testid="collapsedControl"] button span::after {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-repeat: no-repeat;
        background-position: center;
        pointer-events: none;
    }

    /* Chevron Left for Sidebar Collapse */
    [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"]::after,
    [data-testid="stSidebarCollapseButton"] button span::after {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2364648c' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 18 9 12 15 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    }

    /* Chevron Right for Collapsed Control and Expanders */
    [data-testid="collapsedControl"] button [data-testid="stIconMaterial"]::after,
    [data-testid="collapsedControl"] button span::after,
    [data-testid="stExpander"] [data-testid="stIconMaterial"]::after {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2364648c' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='9 18 15 12 9 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    }
    /* Hide any raw text in the collapse button area */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        overflow: hidden !important;
        max-height: 48px !important;
    }
    /* Force the sidebar collapse icon to use SVG fallback */
    [data-testid="stSidebarCollapseButton"] button::before,
    [data-testid="collapsedControl"] button::before {
        content: "" !important;
    }
    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="collapsedControl"] button svg {
        width: 20px !important;
        height: 20px !important;
    }
    /* Ensure no text overflow from the sidebar nav button area */
    [data-testid="stSidebarNavLink"] span,
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="collapsedControl"] * {
        text-overflow: ellipsis !important;
        overflow: hidden !important;
        white-space: nowrap !important;
    }

    :root {
        --bg-primary: #f8f9fc;
        --bg-secondary: #ffffff;
        --bg-tertiary: #f0f2f8;
        --bg-card: rgba(255, 255, 255, 0.8);
        --bg-glass: rgba(255, 255, 255, 0.6);
        --bg-glass-hover: rgba(255, 255, 255, 0.8);
        --bg-hover: rgba(99, 102, 241, 0.06);
        --text-primary: #1a1a2e;
        --text-secondary: #64648c;
        --text-muted: #9898b0;
        --accent: #6366f1;
        --accent-light: #4f46e5;
        --accent-glow: rgba(99, 102, 241, 0.15);
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        --border: rgba(0, 0, 0, 0.08);
        --border-light: rgba(0, 0, 0, 0.12);
        --success: #16a34a;
        --success-bg: rgba(22, 163, 74, 0.08);
        --warning: #d97706;
        --warning-bg: rgba(217, 119, 6, 0.08);
        --error: #dc2626;
        --error-bg: rgba(220, 38, 38, 0.08);
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
        --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.1);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    @keyframes borderGlow {
        0%, 100% { border-color: rgba(99, 102, 241, 0.3); }
        50% { border-color: rgba(99, 102, 241, 0.6); }
    }

    @keyframes slideInUp {
        from { transform: translateY(100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

    .stApp { animation: fadeInUp 0.5s ease-out; }
    h1 { animation: fadeInDown 0.5s ease-out; }
    h2 { animation: fadeInDown 0.4s ease-out 0.05s both; }

    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
    }

    [data-testid="stHeader"] {
        background: var(--bg-primary) !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    .nav-header {
        color: var(--text-muted) !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        padding: 1rem 0.5rem 0.4rem 0.9rem !important;
    }

    .nav-item-active {
        background: var(--accent-glow) !important;
        border-left: 3px solid var(--accent) !important;
        color: var(--accent) !important;
        padding: 0.6rem 0.9rem !important;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        animation: fadeInLeft 0.3s ease-out;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: var(--radius-sm) !important;
        transition: all var(--transition-normal) !important;
        font-size: 0.9rem !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
        transform: translateX(4px) !important;
    }

    [data-testid="stAppViewContainer"] .stButton > button {
        border-radius: var(--radius-md) !important;
        transition: all var(--transition-normal) !important;
        font-weight: 500 !important;
    }

    [data-testid="stAppViewContainer"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
    [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"] {
        background: var(--accent-gradient) !important;
        background-size: 200% 200% !important;
        animation: gradientShift 3s ease infinite !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px var(--accent-glow) !important;
    }

    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover,
    [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"]:hover {
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-3px) !important;
    }

    /* Secondary buttons */
    [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
        background: var(--bg-glass) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }

    [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
        background: var(--bg-glass-hover) !important;
        border-color: var(--accent) !important;
        color: var(--accent-light) !important;
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        transition: all var(--transition-normal) !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    [data-testid="stMetric"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem !important;
        transition: all var(--transition-normal) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--accent) !important;
        box-shadow: var(--shadow-glow) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stMetricValue"] { color: var(--accent) !important; font-weight: 700 !important; }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        transition: all var(--transition-normal) !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        border-radius: var(--radius-md) !important;
        transition: all var(--transition-normal) !important;
        font-weight: 500 !important;
    }

    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
    }

    /* ─── File Uploader ─── */
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploader"] section {
        border: 2px dashed var(--border-light) !important;
        border-radius: var(--radius-lg) !important;
        transition: all var(--transition-normal) !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--accent) !important;
        animation: borderGlow 2s ease infinite;
    }

    /* ─── Chat Messages ─── */
    [data-testid="stChatMessage"] {
        border-radius: var(--radius-lg) !important;
        animation: fadeInUp 0.3s ease-out !important;
    }

    .stAlert { border-radius: var(--radius-md) !important; }

    /* ─── Progress Bar ─── */
    .stProgress > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 20px !important;
    }

    /* ─── DataFrame ─── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }

    .page-desc { color: var(--text-secondary) !important; font-size: 0.95rem; animation: fadeInUp 0.4s ease-out 0.15s both; }
    .divider { height: 1px; background: var(--border); margin: 0.75rem 0; }

    .welcome-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: var(--radius-xl);
        padding: 2rem;
        text-align: center;
        animation: scaleIn 0.5s ease-out;
    }

    .step-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.2rem;
        transition: all var(--transition-normal);
        box-shadow: var(--shadow-sm);
    }

    .step-card:hover {
        border-color: var(--accent);
        transform: translateY(-3px);
        box-shadow: var(--shadow-glow);
    }

    .step-number {
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; border-radius: 50%;
        background: var(--accent-gradient); color: white;
        font-weight: 700; font-size: 0.85rem; margin-right: 10px;
    }

    .help-tip {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.06) 0%, rgba(139, 92, 246, 0.03) 100%);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        animation: fadeInUp 0.3s ease-out;
    }

    .kpi-card {
        background: var(--bg-secondary); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 1.5rem; text-align: center;
        box-shadow: var(--shadow-sm); transition: all var(--transition-normal);
    }
    .kpi-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-glow); border-color: var(--accent); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
    .kpi-label { color: var(--text-secondary); font-size: 0.85rem; font-weight: 500; }

    .data-info-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.06) 0%, rgba(139, 92, 246, 0.03) 100%);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: var(--radius-md);
        padding: 0.8rem;
    }

    .guided-banner {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.06) 0%, rgba(99, 102, 241, 0.06) 100%);
        border: 1px solid rgba(22, 163, 74, 0.12);
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        animation: fadeInDown 0.3s ease-out;
    }

    .status-indicator { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
    .status-saved { background: var(--success-bg); color: var(--success); border: 1px solid rgba(22, 163, 74, 0.15); }
    .status-saving { background: var(--warning-bg); color: var(--warning); animation: pulse 1.5s infinite; }

    .stColumn:nth-child(1) { animation: fadeInUp 0.3s ease-out 0.05s both; }
    .stColumn:nth-child(2) { animation: fadeInUp 0.3s ease-out 0.1s both; }
    .stColumn:nth-child(3) { animation: fadeInUp 0.3s ease-out 0.15s both; }
    .stColumn:nth-child(4) { animation: fadeInUp 0.3s ease-out 0.2s both; }
    .stColumn:nth-child(5) { animation: fadeInUp 0.3s ease-out 0.25s both; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ─── Spinner ─── */
    .stSpinner > div { border-color: var(--accent) !important; }

    /* ─── Slider ─── */
    .stSlider > div > div > div > div { background: var(--accent) !important; }

    /* ─── Tooltip ─── */
    [data-testid="stTooltipIcon"] { color: var(--accent) !important; }

    /* ─── Notification Toast ─── */
    .toast-notification {
        position: fixed; top: 80px; right: 20px;
        padding: 12px 20px; border-radius: var(--radius-md);
        font-size: 0.85rem; z-index: 1001;
        animation: slideInUp 0.3s ease-out;
        max-width: 350px;
    }
    .toast-success { background: var(--success-bg); border: 1px solid rgba(22, 163, 74, 0.3); color: var(--success); }
    .toast-info { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.3); color: var(--accent); }

    /* ─── Dropdown & Popover Fixes ─── */
    /* Ensure dropdown menus have solid background and no transparency bleed-through */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="select"] ul,
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: var(--bg-secondary) !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        box-shadow: var(--shadow-lg) !important;
        border: 1px solid var(--border) !important;
    }

    /* Ensure options are visible and have contrast */
    li[data-baseweb="option"],
    li[role="option"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    li[data-baseweb="option"]:hover,
    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: var(--bg-hover) !important;
        color: var(--accent) !important;
    }

    /* Ensure select input background is solid */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-secondary) !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Apply theme
apply_theme()

# ─── Authentication ──────────────────────────────────────────────────
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
        st.markdown("""
        <div class="welcome-banner">
            <h2 style="margin-bottom: 0.5rem;">🔬 Data Engine</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">AI-Powered Data Science Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        with st.form("login_form"):
            entered_password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter your password...")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submitted:
                if entered_password == password:
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
    return False

if not check_password():
    st.stop()

# ─── Import Components ───────────────────────────────────────────────
from components import ingestion, profiling, cleaning, transformation, analysis, visualization, insights, chat, reporting
from components import modeling, timeseries, feature_engineering, advanced_analysis, dashboard, workflow
from components import groq_agent
from utils.db import (
    init_db, save_project, load_projects, load_project_details,
    load_uploaded_files_list, load_uploaded_file_data,
    save_user_session, save_analysis_step, load_user_preferences,
    save_user_preferences, auto_save_data
)

# Initialize database
init_db()

# Load user preferences
try:
    prefs = load_user_preferences("default_user")
    if prefs:
        st.session_state['user_preferences'] = prefs
        if prefs.get('dark_mode') is not None:
            st.session_state['dark_mode'] = prefs['dark_mode']
        if prefs.get('guided_mode') is not None:
            st.session_state['guided_mode'] = prefs['guided_mode']
        if prefs.get('onboarding_complete') is not None:
            st.session_state['onboarding_complete'] = prefs['onboarding_complete']
except Exception:
    pass

# Auto-load most recent file from database if no data in session
if st.session_state.get('data') is None:
    recent_files = load_uploaded_files_list()
    if recent_files:
        latest = recent_files[0]
        df, name, ftype = load_uploaded_file_data(latest['id'])
        if df is not None:
            st.session_state['data'] = df
            st.session_state['file_meta'] = {
                'name': name,
                'size': latest.get('file_size', 0),
                'type': ftype,
                'source': latest.get('source', 'database')
            }


# ─── Page Configuration ──────────────────────────────────────────────
PAGE_GROUPS = {
    "GET STARTED": [
        "Data Ingestion",
    ],
    "EXPLORE & PREPARE": [
        "Data Profiling", "Data Cleaning", "Transformation",
    ],
    "VISUALIZE": [
        "Visualization", "Dashboard",
    ],
    "ANALYZE": [
        "Statistical Analysis", "Feature Engineering", "Predictive Modeling",
        "Time Series", "Advanced Analysis",
    ],
    "AI & AUTOMATION": [
        "AI Insights", "Chat", "Workflow Automation",
    ],
    "EXPORT": [
        "Reporting",
    ],
}

PAGES = []
for group_pages in PAGE_GROUPS.values():
    PAGES.extend(group_pages)

PAGE_ICONS = {
    "Data Ingestion": "📤", "Data Profiling": "📊", "Data Cleaning": "🧹", "Transformation": "🔄",
    "Feature Engineering": "⚙️", "Predictive Modeling": "🎯", "Time Series": "📈", "Advanced Analysis": "🧠",
    "Statistical Analysis": "📉", "Visualization": "📊", "Dashboard": "🖥️", "AI Insights": "💡", "Chat": "💬",
    "Reporting": "📄", "Workflow Automation": "⚡"
}

PAGE_DESC = {
    "Data Ingestion": "Upload your data from files, URLs, APIs, or pick a sample dataset to get started",
    "Data Profiling": "Get a complete picture of your data - quality scores, distributions, and patterns",
    "Data Cleaning": "Fix missing values, remove duplicates, and get your data ready for analysis",
    "Transformation": "Filter, aggregate, pivot, merge, and reshape your data as needed",
    "Feature Engineering": "Create new features from your data to improve analysis and predictions",
    "Predictive Modeling": "Build and compare machine learning models to make predictions",
    "Time Series": "Analyze trends over time and forecast future values",
    "Advanced Analysis": "Run advanced techniques like PCA, clustering, and anomaly detection",
    "Statistical Analysis": "Perform statistical tests, clustering, and outlier analysis",
    "Visualization": "Create beautiful interactive charts and discover visual patterns",
    "Dashboard": "Build your custom dashboard with KPIs and pinned visualizations",
    "AI Insights": "Let AI analyze your data and provide comprehensive recommendations",
    "Chat": "Ask questions about your data in plain English and get instant answers",
    "Reporting": "Generate professional reports and export your data in any format",
    "Workflow Automation": "Automate your data pipeline from cleaning to modeling in one click"
}

# Friendly descriptions for non-technical users
PAGE_HELP = {
    "Data Ingestion": "This is where you bring your data in. You can upload a file from your computer (like Excel or CSV), paste a web link, or try one of our sample datasets to explore the app.",
    "Data Profiling": "Think of this as a health check for your data. It tells you how many rows and columns you have, finds missing information, spots unusual values, and gives your data a quality score.",
    "Data Cleaning": "Your data might have gaps or errors. This tool helps you fill in missing information, remove duplicate entries, and standardize everything so it's ready for analysis.",
    "Transformation": "Need to reshape your data? Here you can filter rows, combine datasets, create new calculated columns, and organize your data in different ways.",
    "Feature Engineering": "This creates new data points from your existing data that can help with predictions. For example, extracting the month from a date, or combining two columns.",
    "Predictive Modeling": "This is where the magic happens! Train AI models to make predictions. Pick what you want to predict, choose an algorithm, and the app does the rest.",
    "Time Series": "If your data has dates or times, use this to find trends, seasonal patterns, and forecast what might happen next.",
    "Advanced Analysis": "Advanced tools for finding hidden patterns, detecting unusual data points, and reducing complex data into simpler forms.",
    "Statistical Analysis": "Run statistical tests to validate your hypotheses, find clusters in your data, and detect outliers scientifically.",
    "Visualization": "Create beautiful charts and graphs. The AI can even suggest the best chart types for your data!",
    "Dashboard": "Pin your favorite charts and KPIs to create a custom dashboard you can share or monitor.",
    "AI Insights": "Our AI reads through your entire dataset and writes a comprehensive analysis with key findings and recommendations.",
    "Chat": "Just type a question about your data in plain English. The AI will analyze it and give you an answer - no coding required!",
    "Reporting": "Export everything into a professional report (HTML, PDF, or Word) or download your processed data.",
    "Workflow Automation": "Set up an automated pipeline that cleans your data, engineers features, and builds models - all with one click."
}

# Suggested next steps for each page
PAGE_NEXT_STEPS = {
    "Data Ingestion": ["Data Profiling", "Visualization"],
    "Data Profiling": ["Data Cleaning", "Visualization", "AI Insights"],
    "Data Cleaning": ["Data Profiling", "Transformation", "Feature Engineering"],
    "Transformation": ["Visualization", "Feature Engineering", "Statistical Analysis"],
    "Feature Engineering": ["Predictive Modeling", "Visualization"],
    "Predictive Modeling": ["Visualization", "Reporting", "AI Insights"],
    "Time Series": ["Visualization", "Reporting", "AI Insights"],
    "Advanced Analysis": ["Visualization", "Reporting"],
    "Statistical Analysis": ["Visualization", "Advanced Analysis", "Reporting"],
    "Visualization": ["Dashboard", "Reporting"],
    "Dashboard": ["Reporting"],
    "AI Insights": ["Chat", "Reporting", "Dashboard"],
    "Chat": ["AI Insights", "Visualization"],
    "Reporting": [],
    "Workflow Automation": ["Dashboard", "Reporting"],
}


# ─── Helper Functions ────────────────────────────────────────────────
def add_notification(message, notif_type="info"):
    """Add a notification to the session."""
    st.session_state['notifications'].append({
        'message': message,
        'type': notif_type,
        'time': datetime.now().strftime("%H:%M")
    })

def mark_step_complete(step_name):
    """Mark a workflow step as completed."""
    st.session_state['completed_steps'].add(step_name)
    try:
        save_analysis_step("default_user", step_name, {"completed": True})
    except Exception:
        pass

def get_progress_percentage():
    """Calculate overall workflow progress."""
    key_steps = ["Data Ingestion", "Data Profiling", "Data Cleaning", "Visualization", "AI Insights"]
    completed = sum(1 for s in key_steps if s in st.session_state.get('completed_steps', set()))
    return int((completed / len(key_steps)) * 100)

def perform_auto_save():
    """Auto-save current data state to database."""
    if st.session_state.get('data') is not None and st.session_state.get('file_meta'):
        try:
            auto_save_data(
                st.session_state['data'],
                st.session_state['file_meta'].get('name', 'auto_save'),
                st.session_state['file_meta'].get('type', 'csv')
            )
            st.session_state['auto_save_status'] = 'saved'
            st.session_state['last_saved_at'] = datetime.now().strftime("%H:%M:%S")
        except Exception:
            st.session_state['auto_save_status'] = 'error'


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("""
    <div style="padding: 0.5rem 0; text-align: center;">
        <h2 style="margin: 0; background: linear-gradient(135deg, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.4rem;">🔬 Data Engine</h2>
        <p style="color: var(--text-muted); font-size: 0.7rem; margin: 2px 0 0 0; letter-spacing: 0.1em;">AI-POWERED DATA SCIENCE</p>
    </div>
    """, unsafe_allow_html=True)

    # Mode Toggle Row (theme toggle removed - light mode only)
    tc1, tc2 = st.columns([1, 1])
    with tc1:
        guide_icon = "🎯" if st.session_state.get('guided_mode') else "⚡"
        if st.button(guide_icon, help="Toggle guided mode (tips & suggestions)", key="guide_toggle"):
            st.session_state['guided_mode'] = not st.session_state.get('guided_mode', True)
            try:
                save_user_preferences("default_user", {
                    **st.session_state.get('user_preferences', {}),
                    'guided_mode': st.session_state['guided_mode']
                })
            except Exception:
                pass
            st.rerun()
    with tc2:
        if st.button("❓", help="Show help for current page", key="help_toggle"):
            st.session_state['show_help_panel'] = not st.session_state.get('show_help_panel', False)
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Data Status Card
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        file_name = st.session_state['file_meta']['name']
        if len(file_name) > 22:
            file_name = file_name[:19] + "..."

        st.markdown(f"""
        <div class="data-info-card">
            <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 4px;">📁 {file_name}</div>
            <div style="display: flex; gap: 12px; font-size: 0.75rem; color: var(--text-secondary);">
                <span>{len(df):,} rows</span>
                <span>·</span>
                <span>{len(df.columns)} cols</span>
                <span>·</span>
                <span>{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Auto-save status
        if st.session_state.get('last_saved_at'):
            status_class = "status-saved" if st.session_state.get('auto_save_status') == 'saved' else "status-error"
            status_icon = "✓" if st.session_state.get('auto_save_status') == 'saved' else "✗"
            st.markdown(f"""
            <div class="status-indicator {status_class}" style="margin-top: 4px;">
                {status_icon} Saved at {st.session_state['last_saved_at']}
            </div>
            """, unsafe_allow_html=True)

        # Progress tracker
        progress = get_progress_percentage()
        if progress > 0 and progress < 100:
            st.progress(progress / 100, text=f"Workflow: {progress}% complete")
    else:
        st.markdown("""
        <div class="data-info-card" style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 4px;">📂</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">No data loaded yet</div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 2px;">Upload a file or pick a sample dataset</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Navigation with groups
    for group_name, group_pages in PAGE_GROUPS.items():
        st.markdown(f'<p class="nav-header">{group_name}</p>', unsafe_allow_html=True)

        for page in group_pages:
            icon = PAGE_ICONS.get(page, "")
            is_active = st.session_state.get('current_page') == page
            is_completed = page in st.session_state.get('completed_steps', set())

            if is_active:
                status_mark = " ✓" if is_completed else ""
                st.markdown(f'<div class="nav-item-active">{icon} {page}{status_mark}</div>', unsafe_allow_html=True)
            else:
                label = f"{icon} {page}"
                if is_completed:
                    label += " ✓"
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    st.session_state['current_page'] = page
                    st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        st.caption("**Project Management**")
        proj_name = st.text_input("Name", label_visibility="collapsed", placeholder="Project name...")
        if st.button("💾 Save Project", use_container_width=True):
            if st.session_state['data'] is not None and proj_name:
                summary = {"rows": len(st.session_state['data']), "cols": len(st.session_state['data'].columns)}
                if save_project(proj_name, summary, st.session_state.get('last_insights', '')):
                    st.success("Project saved!")
                    add_notification(f"Project '{proj_name}' saved", "success")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.caption("**Preferences**")
        if st.button("🔄 Reset Onboarding", use_container_width=True):
            st.session_state['onboarding_complete'] = False
            st.session_state['completed_steps'] = set()
            try:
                save_user_preferences("default_user", {
                    **st.session_state.get('user_preferences', {}),
                    'onboarding_complete': False
                })
            except Exception:
                pass
            st.rerun()

    # Version
    mode_label = "Guided" if st.session_state.get('guided_mode') else "Expert"
    st.caption(f"v{config['version']} · {mode_label}")


# ─── Main Content ────────────────────────────────────────────────────
page = st.session_state.get('current_page', "Data Ingestion")
icon = PAGE_ICONS.get(page, "")
desc = PAGE_DESC.get(page, "")

# ─── Onboarding Experience ───────────────────────────────────────────
if not st.session_state.get('onboarding_complete', False) and page == "Data Ingestion" and st.session_state.get('data') is None:
    st.markdown("""
    <div class="welcome-banner">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">Welcome to Data Engine! 🚀</h1>
        <p style="color: var(--text-secondary); font-size: 1.05rem; max-width: 600px; margin: 0 auto 1rem;">
            Your AI-powered data analysis assistant. No coding or data science experience needed.
            <br>Let's walk through what you can do.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Step-by-step guide
    cols = st.columns(3)
    steps = [
        ("1", "Upload Your Data", "Start by uploading a CSV, Excel, or JSON file. Or try a sample dataset to explore!", "📤"),
        ("2", "Explore & Clean", "See stats, find issues, and clean your data with one-click fixes.", "🔍"),
        ("3", "Visualize & Analyze", "Create charts, build dashboards, train AI models, and export reports.", "📊"),
    ]

    for i, (num, title, desc_text, emoji) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="step-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                <div><span class="step-number">{num}</span><strong>{title}</strong></div>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">{desc_text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Got it! Let's get started", type="primary", use_container_width=True):
            st.session_state['onboarding_complete'] = True
            try:
                save_user_preferences("default_user", {
                    **st.session_state.get('user_preferences', {}),
                    'onboarding_complete': True
                })
            except Exception:
                pass
            st.rerun()
        if st.button("Skip intro (I know what I'm doing)", use_container_width=True):
            st.session_state['onboarding_complete'] = True
            st.session_state['guided_mode'] = False
            try:
                save_user_preferences("default_user", {
                    'onboarding_complete': True,
                    'guided_mode': False
                })
            except Exception:
                pass
            st.rerun()

    st.markdown("---")

# Page header
st.markdown(f"# {icon} {page}")
st.markdown(f'<p class="page-desc">{desc}</p>', unsafe_allow_html=True)

# Contextual help panel
if st.session_state.get('show_help_panel', False):
    help_text = PAGE_HELP.get(page, "No help available for this page.")
    st.markdown(f"""
    <div class="help-tip">
        <strong>💡 What is this page?</strong><br>
        {help_text}
    </div>
    """, unsafe_allow_html=True)

# Guided mode tips
if st.session_state.get('guided_mode', True) and st.session_state.get('data') is None and page != "Data Ingestion":
    st.markdown("""
    <div class="guided-banner">
        💡 <strong>Tip:</strong> You need to upload data first before using this feature. Go to <strong>Data Ingestion</strong> to upload a file or pick a sample dataset.
    </div>
    """, unsafe_allow_html=True)

# ─── Render Page ─────────────────────────────────────────────────────
if page == "Data Ingestion":
    ingestion.render()
    if st.session_state.get('data') is not None:
        mark_step_complete("Data Ingestion")
        perform_auto_save()
elif page == "Data Profiling":
    profiling.render()
    if st.session_state.get('data') is not None:
        mark_step_complete("Data Profiling")
elif page == "Data Cleaning":
    cleaning.render()
    if st.session_state.get('data') is not None:
        perform_auto_save()
        mark_step_complete("Data Cleaning")
elif page == "Transformation":
    transformation.render()
    if st.session_state.get('data') is not None:
        perform_auto_save()
elif page == "Feature Engineering":
    feature_engineering.render()
elif page == "Predictive Modeling":
    modeling.render()
elif page == "Time Series":
    timeseries.render()
elif page == "Advanced Analysis":
    advanced_analysis.render()
elif page == "Statistical Analysis":
    analysis.render()
elif page == "Visualization":
    visualization.render()
    if st.session_state.get('data') is not None:
        mark_step_complete("Visualization")
elif page == "Dashboard":
    dashboard.render()
elif page == "AI Insights":
    insights.render()
    if st.session_state.get('last_insights'):
        mark_step_complete("AI Insights")
elif page == "Chat":
    chat.render()
elif page == "Reporting":
    reporting.render()
elif page == "Workflow Automation":
    workflow.render()

# ─── Next Steps Suggestion (Guided Mode) ─────────────────────────────
if st.session_state.get('guided_mode', True) and st.session_state.get('data') is not None:
    next_steps = PAGE_NEXT_STEPS.get(page, [])
    if next_steps:
        st.markdown("---")
        st.markdown("### 🧭 What's Next?")
        st.markdown('<p style="color: var(--text-secondary); font-size: 0.9rem;">Based on where you are, here are some suggested next steps:</p>', unsafe_allow_html=True)

        next_cols = st.columns(min(len(next_steps), 3))
        for i, next_page in enumerate(next_steps[:3]):
            with next_cols[i]:
                next_icon = PAGE_ICONS.get(next_page, "")
                next_desc = PAGE_DESC.get(next_page, "")
                is_done = next_page in st.session_state.get('completed_steps', set())
                status_text = " ✓ Done" if is_done else ""

                st.markdown(f"""
                <div class="step-card">
                    <div style="font-size: 1.2rem;">{next_icon}</div>
                    <strong>{next_page}{status_text}</strong>
                    <p style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 4px;">{next_desc[:80]}...</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Go to {next_page}", key=f"next_{next_page}", use_container_width=True):
                    st.session_state['current_page'] = next_page
                    st.rerun()

# ─── Groq AI Assistant ───────────────────────────────────────────────
groq_agent.render_agent(page)
