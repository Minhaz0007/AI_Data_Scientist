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
    st.session_state['dark_mode'] = True  # Default to dark mode for modern look
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
    if st.session_state.get('dark_mode', True):
        st.markdown("""
        <style>
        /* ═══════════════════════════════════════════════════════════
           MODERN DARK THEME - Glassmorphism + Smooth Animations
           ═══════════════════════════════════════════════════════════ */

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a2e;
            --bg-card: rgba(26, 26, 46, 0.6);
            --bg-glass: rgba(255, 255, 255, 0.03);
            --bg-glass-hover: rgba(255, 255, 255, 0.06);
            --bg-hover: rgba(99, 102, 241, 0.1);
            --text-primary: #f0f0f5;
            --text-secondary: #a0a0b8;
            --text-muted: #6b6b80;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
            --border: rgba(255, 255, 255, 0.08);
            --border-light: rgba(255, 255, 255, 0.12);
            --success: #22c55e;
            --success-bg: rgba(34, 197, 94, 0.1);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.1);
            --error: #ef4444;
            --error-bg: rgba(239, 68, 68, 0.1);
            --info: #6366f1;
            --info-bg: rgba(99, 102, 241, 0.1);
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-normal: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ─── Keyframe Animations ─── */
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

        @keyframes fadeInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideInUp {
            from { transform: translateY(100%); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes scaleIn {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
        }

        @keyframes ripple {
            0% { transform: scale(0); opacity: 1; }
            100% { transform: scale(4); opacity: 0; }
        }

        @keyframes borderGlow {
            0%, 100% { border-color: rgba(99, 102, 241, 0.3); }
            50% { border-color: rgba(99, 102, 241, 0.6); }
        }

        @keyframes successPop {
            0% { transform: scale(0.8); opacity: 0; }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); opacity: 1; }
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        /* ─── Base App ─── */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: var(--bg-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        .stApp {
            animation: fadeInUp 0.5s ease-out;
        }

        [data-testid="stAppViewContainer"] > .main {
            animation: fadeInUp 0.4s ease-out 0.1s both;
        }

        /* ─── Sidebar ─── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, #0d0d15 100%) !important;
            border-right: 1px solid var(--border) !important;
            transition: all var(--transition-normal);
        }

        section[data-testid="stSidebar"] > div {
            background: transparent !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border: none !important;
            text-align: left !important;
            padding: 0.6rem 0.9rem !important;
            font-weight: 400 !important;
            font-size: 0.9rem !important;
            justify-content: flex-start !important;
            border-radius: var(--radius-sm) !important;
            transition: all var(--transition-normal) !important;
            position: relative;
            overflow: hidden;
            width: 100% !important;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background: var(--bg-hover) !important;
            color: var(--text-primary) !important;
            transform: translateX(4px) !important;
            padding-left: 1.1rem !important;
        }

        section[data-testid="stSidebar"] .stButton > button:active {
            transform: translateX(2px) scale(0.98) !important;
        }

        /* ─── Typography ─── */
        .stMarkdown, p, span, label, li, div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        .stMarkdown p, p, span, label, li {
            color: var(--text-primary) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        h1 { animation: fadeInDown 0.5s ease-out; }
        h2 { animation: fadeInDown 0.4s ease-out 0.05s both; }
        h3 { animation: fadeInDown 0.3s ease-out 0.1s both; }

        /* ─── Navigation Styles ─── */
        .nav-header {
            color: var(--text-muted) !important;
            font-size: 0.65rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.15em !important;
            padding: 1rem 0.5rem 0.4rem 0.9rem !important;
            margin: 0 !important;
        }

        .nav-item-active {
            background: var(--accent-glow) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--accent-light) !important;
            padding: 0.6rem 0.9rem !important;
            margin: 2px 0 !important;
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            animation: fadeInLeft 0.3s ease-out;
            backdrop-filter: blur(10px);
            box-shadow: inset 0 0 20px rgba(99, 102, 241, 0.1);
        }

        /* ─── Cards & Containers ─── */
        .glass-card {
            background: var(--bg-card) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow-md) !important;
            transition: all var(--transition-normal) !important;
            animation: scaleIn 0.4s ease-out;
        }

        .glass-card:hover {
            border-color: var(--border-light) !important;
            box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
            transform: translateY(-2px) !important;
        }

        /* ─── Buttons ─── */
        [data-testid="stAppViewContainer"] .stButton > button {
            border-radius: var(--radius-md) !important;
            transition: all var(--transition-normal) !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            position: relative;
            overflow: hidden;
        }

        [data-testid="stAppViewContainer"] .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-md) !important;
        }

        [data-testid="stAppViewContainer"] .stButton > button:active {
            transform: translateY(0px) scale(0.98) !important;
        }

        /* Primary buttons */
        [data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
        [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"] {
            background: var(--accent-gradient) !important;
            background-size: 200% 200% !important;
            animation: gradientShift 3s ease infinite !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em;
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
            backdrop-filter: blur(10px) !important;
        }

        [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
            background: var(--bg-glass-hover) !important;
            border-color: var(--accent) !important;
            color: var(--accent-light) !important;
        }

        /* ─── Input Fields ─── */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stMultiSelect > div > div > div {
            background: var(--bg-glass) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            transition: all var(--transition-normal) !important;
            backdrop-filter: blur(10px) !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stSelectbox > div > div > div[data-baseweb="select"] > div {
            color: var(--text-primary) !important;
        }

        ul[data-baseweb="menu"] li {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            transition: all var(--transition-fast) !important;
        }

        ul[data-baseweb="menu"] li:hover {
            background-color: var(--bg-hover) !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-glow) !important;
        }

        /* ─── Chat Input ─── */
        [data-testid="stChatInput"] {
            border-radius: var(--radius-lg) !important;
        }

        [data-testid="stChatInput"] textarea {
            background: var(--bg-glass) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            color: var(--text-primary) !important;
            transition: all var(--transition-normal) !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-glow) !important;
        }

        /* ─── Chat Messages ─── */
        [data-testid="stChatMessage"] {
            background: var(--bg-glass) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1rem !important;
            animation: fadeInUp 0.3s ease-out !important;
            transition: all var(--transition-normal) !important;
        }

        [data-testid="stChatMessage"]:hover {
            border-color: var(--border-light) !important;
        }

        /* ─── DataFrame ─── */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
            animation: fadeInUp 0.4s ease-out;
        }

        /* ─── Metrics ─── */
        [data-testid="stMetric"] {
            background: var(--bg-glass) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem !important;
            transition: all var(--transition-normal) !important;
        }

        [data-testid="stMetric"]:hover {
            border-color: var(--accent) !important;
            box-shadow: var(--shadow-glow) !important;
            transform: translateY(-2px) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--accent-light) !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }

        /* ─── Tabs ─── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 1px solid var(--border) !important;
            gap: 0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
            padding: 0.75rem 1.2rem !important;
            transition: all var(--transition-normal) !important;
            border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
            font-weight: 500 !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary) !important;
            background: var(--bg-glass) !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent-light) !important;
            border-bottom: 2px solid var(--accent) !important;
            background: var(--bg-glass) !important;
        }

        /* ─── Expander ─── */
        .streamlit-expanderHeader {
            background: var(--bg-glass) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            transition: all var(--transition-normal) !important;
            font-weight: 500 !important;
        }

        .streamlit-expanderHeader:hover {
            border-color: var(--accent) !important;
            background: var(--bg-glass-hover) !important;
        }

        details[open] .streamlit-expanderHeader {
            border-color: var(--accent) !important;
        }

        .streamlit-expanderContent {
            border: 1px solid var(--border) !important;
            border-top: none !important;
            border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
            background: var(--bg-glass) !important;
        }

        /* ─── File Uploader ─── */
        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploader"] section {
            background: var(--bg-glass) !important;
            border: 2px dashed var(--border-light) !important;
            border-radius: var(--radius-lg) !important;
            transition: all var(--transition-normal) !important;
        }

        [data-testid="stFileUploadDropzone"]:hover,
        [data-testid="stFileUploader"] section:hover {
            border-color: var(--accent) !important;
            background: var(--bg-hover) !important;
            animation: borderGlow 2s ease infinite;
        }

        [data-testid="stFileUploadDropzone"] *,
        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploader"] section * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploader"] section small {
            color: var(--text-secondary) !important;
        }

        [data-testid="stFileUploadDropzone"] svg,
        [data-testid="stFileUploader"] section svg {
            stroke: var(--text-secondary) !important;
        }

        [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploader"] section button {
            background: var(--bg-tertiary) !important;
            color: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            transition: all var(--transition-normal) !important;
        }

        [data-testid="stFileUploadDropzone"] button:hover,
        [data-testid="stFileUploader"] section button:hover {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            transform: scale(1.02) !important;
        }

        /* ─── Alert Boxes ─── */
        .stAlert {
            border-radius: var(--radius-md) !important;
            animation: fadeInUp 0.3s ease-out !important;
        }

        [data-testid="stAlert"] > div[role="alert"] {
            border-radius: var(--radius-md) !important;
        }

        /* ─── Progress Bar ─── */
        .stProgress > div > div > div {
            background: var(--accent-gradient) !important;
            border-radius: 20px !important;
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        /* ─── Scrollbar ─── */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* ─── Tooltips ─── */
        [data-testid="stTooltipIcon"] {
            color: var(--accent) !important;
            transition: all var(--transition-fast) !important;
        }

        [data-testid="stTooltipIcon"]:hover {
            color: var(--accent-light) !important;
            transform: scale(1.2) !important;
        }

        /* ─── Custom Components ─── */
        .page-desc {
            color: var(--text-secondary) !important;
            font-size: 0.95rem;
            animation: fadeInUp 0.4s ease-out 0.15s both;
        }

        .divider {
            height: 1px;
            background: var(--border);
            margin: 0.75rem 0;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            animation: fadeInUp 0.3s ease-out;
        }

        .status-saved {
            background: var(--success-bg);
            color: var(--success);
            border: 1px solid rgba(34, 197, 94, 0.2);
        }

        .status-saving {
            background: var(--warning-bg);
            color: var(--warning);
            border: 1px solid rgba(245, 158, 11, 0.2);
            animation: pulse 1.5s ease-in-out infinite;
        }

        .status-error {
            background: var(--error-bg);
            color: var(--error);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        /* ─── Onboarding & Help ─── */
        .welcome-banner {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: var(--radius-xl);
            padding: 2rem;
            text-align: center;
            animation: scaleIn 0.5s ease-out;
        }

        .step-card {
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.2rem;
            transition: all var(--transition-normal);
            cursor: pointer;
        }

        .step-card:hover {
            border-color: var(--accent);
            background: var(--bg-hover);
            transform: translateY(-3px);
            box-shadow: var(--shadow-glow);
        }

        .step-card-completed {
            border-color: var(--success) !important;
            background: var(--success-bg) !important;
        }

        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--accent-gradient);
            color: white;
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 10px;
        }

        .step-number-completed {
            background: var(--success) !important;
        }

        .help-tip {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: var(--radius-md);
            padding: 1rem 1.2rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            color: var(--text-secondary);
            animation: fadeInUp 0.3s ease-out;
        }

        .help-tip strong {
            color: var(--accent-light) !important;
        }

        /* ─── KPI Cards ─── */
        .kpi-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            text-align: center;
            transition: all var(--transition-normal);
            animation: fadeInUp 0.4s ease-out;
        }

        .kpi-card:hover {
            border-color: var(--accent);
            box-shadow: var(--shadow-glow);
            transform: translateY(-3px);
        }

        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .kpi-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 0.25rem;
        }

        .kpi-delta {
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }

        .kpi-delta-positive { color: var(--success); }
        .kpi-delta-negative { color: var(--error); }

        /* ─── Progress Tracker ─── */
        .progress-tracker {
            display: flex;
            justify-content: space-between;
            padding: 1rem 0;
            position: relative;
        }

        .progress-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
            position: relative;
            z-index: 1;
        }

        .progress-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--border);
            transition: all var(--transition-normal);
        }

        .progress-dot-active {
            background: var(--accent);
            box-shadow: 0 0 10px var(--accent-glow);
            animation: pulse 2s ease-in-out infinite;
        }

        .progress-dot-completed {
            background: var(--success);
        }

        /* ─── Auto-save Indicator ─── */
        .autosave-indicator {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            z-index: 1000;
            animation: slideInUp 0.3s ease-out;
            backdrop-filter: blur(20px);
        }

        /* ─── Notification Toast ─── */
        .toast-notification {
            position: fixed;
            top: 80px;
            right: 20px;
            padding: 12px 20px;
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            z-index: 1001;
            animation: slideInUp 0.3s ease-out;
            backdrop-filter: blur(20px);
            max-width: 350px;
        }

        .toast-success {
            background: var(--success-bg);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: var(--success);
        }

        .toast-info {
            background: var(--info-bg);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: var(--accent-light);
        }

        /* ─── Stagger Animation for Lists ─── */
        .stColumn:nth-child(1) { animation: fadeInUp 0.3s ease-out 0.05s both; }
        .stColumn:nth-child(2) { animation: fadeInUp 0.3s ease-out 0.1s both; }
        .stColumn:nth-child(3) { animation: fadeInUp 0.3s ease-out 0.15s both; }
        .stColumn:nth-child(4) { animation: fadeInUp 0.3s ease-out 0.2s both; }
        .stColumn:nth-child(5) { animation: fadeInUp 0.3s ease-out 0.25s both; }

        /* ─── Spinner ─── */
        .stSpinner > div {
            border-color: var(--accent) !important;
        }

        /* ─── Slider ─── */
        .stSlider > div > div > div > div {
            background: var(--accent) !important;
        }

        /* ─── Checkbox & Radio ─── */
        .stCheckbox label span,
        .stRadio label span {
            color: var(--text-primary) !important;
        }

        /* ─── Sidebar data info card ─── */
        .data-info-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: var(--radius-md);
            padding: 0.8rem;
            margin: 0.5rem 0;
        }

        /* ─── Guided Mode Banner ─── */
        .guided-banner {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
            border: 1px solid rgba(34, 197, 94, 0.15);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            animation: fadeInDown 0.3s ease-out;
        }

        </style>
        """, unsafe_allow_html=True)
    else:
        # ─── Light Theme ───
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

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

        [data-testid="stFileUploadDropzone"]:hover {
            border-color: var(--accent) !important;
            animation: borderGlow 2s ease infinite;
        }

        .stAlert { border-radius: var(--radius-md) !important; }

        [data-testid="stChatMessage"] {
            border-radius: var(--radius-lg) !important;
            animation: fadeInUp 0.3s ease-out !important;
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

    # Theme & Mode Toggle Row
    tc1, tc2, tc3 = st.columns([1, 1, 1])
    with tc1:
        theme_btn = "☀️" if st.session_state.get('dark_mode') else "🌙"
        if st.button(theme_btn, help="Toggle light/dark theme", key="theme_toggle"):
            st.session_state['dark_mode'] = not st.session_state.get('dark_mode', True)
            try:
                save_user_preferences("default_user", {
                    **st.session_state.get('user_preferences', {}),
                    'dark_mode': st.session_state['dark_mode']
                })
            except Exception:
                pass
            st.rerun()
    with tc2:
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
    with tc3:
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
    theme_label = "Dark" if st.session_state.get('dark_mode') else "Light"
    st.caption(f"v{config['version']} · {theme_label} · {mode_label}")


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
                    'guided_mode': False,
                    'dark_mode': st.session_state.get('dark_mode', True)
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
