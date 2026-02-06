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
    """Initialize the database with necessary tables."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        with engine.connect() as conn:
            if _is_sqlite():
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        summary TEXT,
                        insights TEXT
                    );
                """))
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
            else:
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
            conn.commit()
        return True
    except Exception as e:
        st.error(f"DB Init Error: {e}")
        return False


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


# ─── Uploaded Files CRUD ────────────────────────────────────────────

def save_uploaded_file(df, file_name, file_type, file_size=0, source='file_upload'):
    """Save an uploaded file's data to the database."""
    conn_str = get_connection_string()
    try:
        engine = get_engine(conn_str)
        csv_data = df.to_csv(index=False)
        column_names = json.dumps(df.columns.tolist())
        with engine.connect() as conn:
            # Check if file with same name already exists - update it
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
