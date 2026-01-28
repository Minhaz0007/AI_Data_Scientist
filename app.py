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
