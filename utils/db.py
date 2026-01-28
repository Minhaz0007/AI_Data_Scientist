import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import streamlit as st
import json

def get_connection_string():
    """Get database connection string from environment variable."""
    return os.environ.get("DATABASE_URL")

def init_db():
    """Initialize the database with necessary tables."""
    conn_str = get_connection_string()
    if not conn_str:
        return False

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary JSONB,
                    insights TEXT
                );
            """))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"DB Init Error: {e}")
        return False

def save_project(name, summary, insights):
    """Save project metadata to database."""
    conn_str = get_connection_string()
    if not conn_str:
        return False

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            # Simple insert
            conn.execute(
                text("INSERT INTO projects (name, summary, insights) VALUES (:name, :summary, :insights)"),
                {"name": name, "summary": json.dumps(summary), "insights": insights}
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False

def load_projects():
    """Load list of projects."""
    conn_str = get_connection_string()
    if not conn_str:
        return []

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, name, created_at FROM projects ORDER BY created_at DESC"))
            return result.fetchall()
    except Exception as e:
        st.error(f"Load Error: {e}")
        return []

def load_project_details(project_id):
    """Load details of a specific project."""
    conn_str = get_connection_string()
    if not conn_str:
        return None

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name, summary, insights FROM projects WHERE id = :id"),
                {"id": project_id}
            )
            return result.fetchone()
    except Exception as e:
        st.error(f"Load Details Error: {e}")
        return None
