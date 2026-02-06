import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import streamlit as st
import json
from io import StringIO, BytesIO
from datetime import datetime

# Default to SQLite if no DATABASE_URL is set
SQLITE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_engine.db")


def get_connection_string():
    """Get database connection string. Falls back to SQLite."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return db_url
    return f"sqlite:///{SQLITE_PATH}"


@st.cache_resource
def get_engine(conn_str):
    """Get cached database engine."""
    return create_engine(conn_str)


def _is_sqlite():
    """Check if we are using SQLite."""
    return get_connection_string().startswith("sqlite")


def init_db():
    """Initialize the database with all necessary tables."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            if _is_sqlite():
                # Projects table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        summary TEXT,
                        insights TEXT
                    );
                """))
                # Uploaded files table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER DEFAULT 0,
                        source TEXT DEFAULT 'file_upload',
                        csv_data TEXT,
                        row_count INTEGER DEFAULT 0,
                        col_count INTEGER DEFAULT 0,
                        column_names TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                # User sessions table - tracks user activity
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        current_page TEXT,
                        session_data TEXT
                    );
                """))
                # User preferences table - persists settings
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL UNIQUE,
                        preferences TEXT NOT NULL DEFAULT '{}',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                # Analysis history table - tracks what users have done
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        step_name TEXT NOT NULL,
                        step_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                # Dashboard configurations table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS dashboard_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        name TEXT NOT NULL,
                        config TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                # Auto-save snapshots table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS auto_saves (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        csv_data TEXT,
                        row_count INTEGER DEFAULT 0,
                        col_count INTEGER DEFAULT 0,
                        column_names TEXT,
                        saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
            else:
                # PostgreSQL tables
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        summary JSONB,
                        insights TEXT
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id SERIAL PRIMARY KEY,
                        file_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER DEFAULT 0,
                        source TEXT DEFAULT 'file_upload',
                        csv_data TEXT,
                        row_count INTEGER DEFAULT 0,
                        col_count INTEGER DEFAULT 0,
                        column_names TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        current_page TEXT,
                        session_data JSONB
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL UNIQUE,
                        preferences JSONB NOT NULL DEFAULT '{}',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        step_name TEXT NOT NULL,
                        step_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS dashboard_configs (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        name TEXT NOT NULL,
                        config JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS auto_saves (
                        id SERIAL PRIMARY KEY,
                        file_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        csv_data TEXT,
                        row_count INTEGER DEFAULT 0,
                        col_count INTEGER DEFAULT 0,
                        column_names TEXT,
                        saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"DB Init Error: {e}")
        return False


# ─── Projects CRUD ────────────────────────────────────────────────────

def save_project(name, summary, insights):
    """Save project metadata to database."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
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
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, name, created_at FROM projects ORDER BY created_at DESC"))
            return result.fetchall()
    except Exception as e:
        st.error(f"Load Error: {e}")
        return []


def load_project_details(project_id):
    """Load details of a specific project."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name, summary, insights FROM projects WHERE id = :id"),
                {"id": project_id}
            )
            return result.fetchone()
    except Exception as e:
        st.error(f"Load Details Error: {e}")
        return None


# ─── Uploaded Files CRUD ──────────────────────────────────────────────

def save_uploaded_file(df, file_name, file_type, file_size=0, source='file_upload'):
    """Save an uploaded file's data to the database."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        csv_data = df.to_csv(index=False)
        column_names = json.dumps(df.columns.tolist())
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT id FROM uploaded_files WHERE file_name = :name"),
                {"name": file_name}
            ).fetchone()

            if existing:
                conn.execute(
                    text("""UPDATE uploaded_files
                            SET csv_data = :csv_data, row_count = :rows, col_count = :cols,
                                column_names = :col_names, file_size = :size, file_type = :ftype,
                                source = :source, uploaded_at = :now
                            WHERE file_name = :name"""),
                    {
                        "csv_data": csv_data, "rows": len(df), "cols": len(df.columns),
                        "col_names": column_names, "size": file_size, "ftype": file_type,
                        "source": source, "now": datetime.utcnow().isoformat(), "name": file_name
                    }
                )
            else:
                conn.execute(
                    text("""INSERT INTO uploaded_files
                            (file_name, file_type, file_size, source, csv_data, row_count, col_count, column_names)
                            VALUES (:name, :ftype, :size, :source, :csv_data, :rows, :cols, :col_names)"""),
                    {
                        "name": file_name, "ftype": file_type, "size": file_size,
                        "source": source, "csv_data": csv_data,
                        "rows": len(df), "cols": len(df.columns), "col_names": column_names
                    }
                )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"File Save Error: {e}")
        return False


def load_uploaded_files_list():
    """Load list of all uploaded files (metadata only)."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, file_name, file_type, file_size, source, row_count, col_count, uploaded_at "
                "FROM uploaded_files ORDER BY uploaded_at DESC"
            ))
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
    except Exception as e:
        return []


def load_uploaded_file_data(file_id):
    """Load a specific uploaded file's data as a DataFrame."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT file_name, file_type, csv_data FROM uploaded_files WHERE id = :id"),
                {"id": file_id}
            )
            row = result.fetchone()
            if row:
                mapping = row._mapping
                csv_data = mapping["csv_data"]
                df = pd.read_csv(StringIO(csv_data))
                return df, mapping["file_name"], mapping["file_type"]
        return None, None, None
    except Exception as e:
        st.error(f"File Load Error: {e}")
        return None, None, None


def delete_uploaded_file(file_id):
    """Delete an uploaded file from the database."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM uploaded_files WHERE id = :id"),
                {"id": file_id}
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"File Delete Error: {e}")
        return False


# ─── Auto-Save Functions ─────────────────────────────────────────────

def auto_save_data(df, file_name, file_type):
    """Auto-save the current working data to the auto_saves table and update uploaded_files."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        csv_data = df.to_csv(index=False)
        column_names = json.dumps(df.columns.tolist())

        with engine.connect() as conn:
            # Update uploaded_files table (main sync)
            existing = conn.execute(
                text("SELECT id FROM uploaded_files WHERE file_name = :name"),
                {"name": file_name}
            ).fetchone()

            if existing:
                conn.execute(
                    text("""UPDATE uploaded_files
                            SET csv_data = :csv_data, row_count = :rows, col_count = :cols,
                                column_names = :col_names, uploaded_at = :now
                            WHERE file_name = :name"""),
                    {
                        "csv_data": csv_data, "rows": len(df), "cols": len(df.columns),
                        "col_names": column_names, "now": datetime.utcnow().isoformat(),
                        "name": file_name
                    }
                )

            # Also save a snapshot to auto_saves (keep latest 5)
            conn.execute(
                text("""INSERT INTO auto_saves
                        (file_name, file_type, csv_data, row_count, col_count, column_names)
                        VALUES (:name, :ftype, :csv_data, :rows, :cols, :col_names)"""),
                {
                    "name": file_name, "ftype": file_type, "csv_data": csv_data,
                    "rows": len(df), "cols": len(df.columns), "col_names": column_names
                }
            )

            # Clean up old snapshots (keep only latest 5)
            if _is_sqlite():
                conn.execute(text("""
                    DELETE FROM auto_saves WHERE id NOT IN (
                        SELECT id FROM auto_saves ORDER BY saved_at DESC LIMIT 5
                    )
                """))
            else:
                conn.execute(text("""
                    DELETE FROM auto_saves WHERE id NOT IN (
                        SELECT id FROM auto_saves ORDER BY saved_at DESC LIMIT 5
                    )
                """))

            conn.commit()
        return True
    except Exception as e:
        return False


def load_latest_auto_save():
    """Load the most recent auto-save."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT file_name, file_type, csv_data, saved_at FROM auto_saves ORDER BY saved_at DESC LIMIT 1"
            ))
            row = result.fetchone()
            if row:
                mapping = row._mapping
                df = pd.read_csv(StringIO(mapping["csv_data"]))
                return df, mapping["file_name"], mapping["file_type"], mapping["saved_at"]
        return None, None, None, None
    except Exception:
        return None, None, None, None


# ─── User Sessions ───────────────────────────────────────────────────

def save_user_session(user_id, current_page=None, session_data=None):
    """Save or update user session."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT id FROM user_sessions WHERE user_id = :uid ORDER BY session_start DESC LIMIT 1"),
                {"uid": user_id}
            ).fetchone()

            data_json = json.dumps(session_data) if session_data else '{}'

            if existing:
                conn.execute(
                    text("""UPDATE user_sessions
                            SET last_active = :now, current_page = :page, session_data = :data
                            WHERE id = :id"""),
                    {"now": datetime.utcnow().isoformat(), "page": current_page, "data": data_json, "id": existing[0]}
                )
            else:
                conn.execute(
                    text("""INSERT INTO user_sessions (user_id, current_page, session_data)
                            VALUES (:uid, :page, :data)"""),
                    {"uid": user_id, "page": current_page, "data": data_json}
                )
            conn.commit()
        return True
    except Exception:
        return False


# ─── User Preferences ────────────────────────────────────────────────

def save_user_preferences(user_id, preferences):
    """Save user preferences (theme, guided mode, etc.)."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        prefs_json = json.dumps(preferences)

        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT id FROM user_preferences WHERE user_id = :uid"),
                {"uid": user_id}
            ).fetchone()

            if existing:
                conn.execute(
                    text("""UPDATE user_preferences
                            SET preferences = :prefs, updated_at = :now
                            WHERE user_id = :uid"""),
                    {"prefs": prefs_json, "now": datetime.utcnow().isoformat(), "uid": user_id}
                )
            else:
                conn.execute(
                    text("""INSERT INTO user_preferences (user_id, preferences)
                            VALUES (:uid, :prefs)"""),
                    {"uid": user_id, "prefs": prefs_json}
                )
            conn.commit()
        return True
    except Exception:
        return False


def load_user_preferences(user_id):
    """Load user preferences."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT preferences FROM user_preferences WHERE user_id = :uid"),
                {"uid": user_id}
            )
            row = result.fetchone()
            if row:
                return json.loads(row[0])
        return None
    except Exception:
        return None


# ─── Analysis History ─────────────────────────────────────────────────

def save_analysis_step(user_id, step_name, step_data=None):
    """Record an analysis step in history."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        data_json = json.dumps(step_data) if step_data else '{}'

        with engine.connect() as conn:
            conn.execute(
                text("""INSERT INTO analysis_history (user_id, step_name, step_data)
                        VALUES (:uid, :step, :data)"""),
                {"uid": user_id, "step": step_name, "data": data_json}
            )
            conn.commit()
        return True
    except Exception:
        return False


def load_analysis_history(user_id, limit=20):
    """Load recent analysis history."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(
                text("""SELECT step_name, step_data, created_at
                        FROM analysis_history WHERE user_id = :uid
                        ORDER BY created_at DESC LIMIT :lim"""),
                {"uid": user_id, "lim": limit}
            )
            rows = result.fetchall()
            return [{"step": r[0], "data": json.loads(r[1]) if r[1] else {}, "time": r[2]} for r in rows]
    except Exception:
        return []


# ─── Dashboard Configs ────────────────────────────────────────────────

def save_dashboard_config(user_id, name, config):
    """Save a dashboard configuration."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        config_json = json.dumps(config)

        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT id FROM dashboard_configs WHERE user_id = :uid AND name = :name"),
                {"uid": user_id, "name": name}
            ).fetchone()

            if existing:
                conn.execute(
                    text("""UPDATE dashboard_configs
                            SET config = :config, updated_at = :now
                            WHERE id = :id"""),
                    {"config": config_json, "now": datetime.utcnow().isoformat(), "id": existing[0]}
                )
            else:
                conn.execute(
                    text("""INSERT INTO dashboard_configs (user_id, name, config)
                            VALUES (:uid, :name, :config)"""),
                    {"uid": user_id, "name": name, "config": config_json}
                )
            conn.commit()
        return True
    except Exception:
        return False


def load_dashboard_configs(user_id):
    """Load all dashboard configurations for a user."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            result = conn.execute(
                text("""SELECT id, name, config, updated_at
                        FROM dashboard_configs WHERE user_id = :uid
                        ORDER BY updated_at DESC"""),
                {"uid": user_id}
            )
            rows = result.fetchall()
            return [{"id": r[0], "name": r[1], "config": json.loads(r[2]), "updated_at": r[3]} for r in rows]
    except Exception:
        return []
