# Deployment Guide: Neon Database & Streamlit

This guide outlines how to deploy the AI Data Analyst app and connect it to a Neon (PostgreSQL) database.

## Important Note on Vercel

While Vercel is excellent for frontend applications, **it is not recommended for Streamlit applications**. Streamlit relies on persistent WebSocket connections and in-memory session state, which are incompatible with Vercel's stateless Serverless Functions architecture.

We recommend deploying to **Streamlit Cloud** (free, easiest) or **Railway/Render** (Docker-based).

## Step 1: Set up Neon Database

1.  **Create a Project**:
    - Log in to your [Neon console](https://neon.tech).
    - Click **"New Project"**.
    - Give it a name (e.g., `ai-data-analyst`).
    - Click **"Create Project"**.

2.  **Get Connection String**:
    - Copy the **Connection String** from the dashboard (e.g., `postgres://user:pass@ep-xyz.neon.tech/neondb?sslmode=require`).
    - **Save this URL**.

## Step 2: Deploy to Streamlit Cloud (Recommended)

Streamlit Cloud is the native hosting platform for Streamlit and supports Neon out of the box.

1.  **Push to GitHub**:
    - Ensure your code is in a GitHub repository.

2.  **Connect to Streamlit Cloud**:
    - Go to [share.streamlit.io](https://share.streamlit.io).
    - Click **"New App"**.
    - Select your GitHub repository, branch, and main file path (`app.py`).

3.  **Configure Secrets (Environment Variables)**:
    - Click **"Advanced Settings"**.
    - In the **Secrets** box, add your configuration:

    ```toml
    # Neon Database Connection
    DATABASE_URL = "postgres://user:pass@ep-xyz.neon.tech/neondb?sslmode=require"

    # Security (Required for web deployment)
    APP_PASSWORD = "your-secure-password"

    # Optional AI Keys
    ANTHROPIC_API_KEY = "sk-..."
    GOOGLE_API_KEY = "AI..."
    ```

4.  **Deploy**:
    - Click **"Deploy!"**.

## Step 3: Initialize the Database

1.  Once deployed, access your app URL.
2.  Enter the `APP_PASSWORD` you set in the secrets.
3.  In the sidebar, click **"Initialize DB (First Run)"**.
4.  You can now save and load projects using your Neon database.
