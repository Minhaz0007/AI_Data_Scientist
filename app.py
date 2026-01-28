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
    layout=config['layout'],
    initial_sidebar_state="expanded"
)

# Authentication Check
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    """Checks the password against the environment variable."""
    password = os.environ.get("APP_PASSWORD")
    if not password:
        # If no password set in env, allow access (dev mode or public if intended)
        return True

    if st.session_state['authenticated']:
        return True

    placeholder = st.empty()
    with placeholder.form("login"):
        st.markdown("### 🔒 Access Required")
        entered_password = st.text_input("Enter App Password", type="password")
        if st.form_submit_button("Login"):
            if entered_password == password:
                st.session_state['authenticated'] = True
                placeholder.empty()
                return True
            else:
                st.error("Incorrect Password")
                return False
    return False

if not check_password():
    st.stop()

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'file_meta' not in st.session_state:
    st.session_state['file_meta'] = None

# Sidebar Navigation
st.sidebar.title(config['app_name'])
page = st.sidebar.radio(
    "Navigate",
    ["Data Ingestion", "Data Profiling", "Data Cleaning", "Transformation",
     "Analysis", "Visualization", "AI Insights", "Chat", "Reporting"]
)

st.title(page)

# Import components
from components import ingestion, profiling, cleaning, transformation, analysis, visualization, insights, chat, reporting
from utils.db import init_db, save_project, load_projects, load_project_details

# Database & Project Management Sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Project Management")

    if os.environ.get("DATABASE_URL"):
        if st.button("Initialize DB (First Run)"):
            if init_db():
                st.success("Database initialized.")

        tab_save, tab_load = st.tabs(["Save", "Load"])

        with tab_save:
            project_name = st.text_input("Project Name")
            if st.button("Save Current Analysis"):
                if st.session_state['data'] is not None:
                    # Create a summary (reuse from insights logic or simple one)
                    summary = {
                        "rows": len(st.session_state['data']),
                        "cols": len(st.session_state['data'].columns)
                    }
                    insights_txt = st.session_state.get('last_insights', "No insights generated.")

                    if save_project(project_name, summary, insights_txt):
                        st.success("Project saved!")
                else:
                    st.warning("No data to save.")

        with tab_load:
            if st.button("Refresh List"):
                projects = load_projects()
                st.session_state['project_list'] = projects

            if 'project_list' in st.session_state:
                opts = {f"{p[1]} ({p[2]})": p[0] for p in st.session_state['project_list']}
                selected_opt = st.selectbox("Select Project", list(opts.keys()))

                if st.button("Load Project"):
                    pid = opts[selected_opt]
                    details = load_project_details(pid)
                    if details:
                        st.info(f"Loaded: {details[0]}")
                        st.text(f"Insights: {details[2]}")
                        # Note: We aren't saving the full DF to DB in this MVP, just metadata/insights.
    else:
        st.info("Configure DATABASE_URL to enable project saving.")

# Component Rendering
if page == "Data Ingestion":
    ingestion.render()
elif page == "Data Profiling":
    profiling.render()
elif page == "Data Cleaning":
    cleaning.render()
elif page == "Transformation":
    transformation.render()
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

st.sidebar.markdown("---")
st.sidebar.info(f"Version {config['version']}")
