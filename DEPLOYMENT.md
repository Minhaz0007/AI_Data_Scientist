# Deployment Guide: AI Data Analyst Application

This guide covers deployment options for the AI Data Analyst app with Neon (PostgreSQL) database integration.

## Important: Vercel Compatibility

**Vercel is NOT recommended for Streamlit applications.**

Streamlit applications require:
- Persistent WebSocket connections for real-time updates
- In-memory session state that persists across requests
- Long-running processes for data analysis

Vercel's serverless architecture:
- Terminates connections after function execution
- Has cold starts that reset state
- Has execution time limits (10-60 seconds)

**Recommended alternatives:**
1. **Streamlit Cloud** (Free, easiest) - Native Streamlit hosting
2. **Railway** (Docker-based, generous free tier)
3. **Render** (Docker-based, free tier available)
4. **Heroku** (Container-based)
5. **DigitalOcean App Platform** (Docker support)

---

## Option 1: Streamlit Cloud (Recommended)

Streamlit Cloud is the native hosting platform and the easiest option.

### Step 1: Push to GitHub

Ensure your code is in a GitHub repository:

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New App"**
3. Select your GitHub repository, branch, and main file (`app.py`)

### Step 3: Configure Secrets

Click **"Advanced Settings"** and add your secrets:

```toml
# .streamlit/secrets.toml format

# Neon Database Connection
DATABASE_URL = "postgres://user:password@ep-xxx.neon.tech/neondb?sslmode=require"

# Security (Required for web deployment)
APP_PASSWORD = "your-secure-password-here"

# Optional: Pre-configure AI API keys
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_API_KEY = "AIza..."
GROQ_API_KEY = "gsk_..."
OPENAI_API_KEY = "sk-..."
```

### Step 4: Deploy

Click **"Deploy!"** and wait for the build to complete.

---

## Option 2: Railway (Docker-based)

Railway supports Docker and is ideal for Streamlit apps.

### Step 1: Create Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF generation
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. Create a new project
3. Select "Deploy from GitHub repo"
4. Add environment variables:
   - `DATABASE_URL`: Your Neon connection string
   - `APP_PASSWORD`: Your app password

### Step 3: Configure Port

Railway will automatically detect the Dockerfile and deploy. Ensure port 8501 is exposed.

---

## Option 3: Render (Docker-based)

### Step 1: Create render.yaml

Create a `render.yaml` file:

```yaml
services:
  - type: web
    name: ai-data-analyst
    env: docker
    plan: free
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: APP_PASSWORD
        sync: false
```

### Step 2: Deploy

1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repo
4. Configure environment variables

---

## Neon Database Setup

### Step 1: Create Neon Project

1. Log in to [Neon Console](https://neon.tech)
2. Click **"New Project"**
3. Name it (e.g., `ai-data-analyst`)
4. Click **"Create Project"**

### Step 2: Get Connection String

Copy the connection string from the dashboard:
```
postgres://user:password@ep-xxx.neon.tech/neondb?sslmode=require
```

### Step 3: Initialize Database

After deploying your app:
1. Access your app URL
2. Enter the APP_PASSWORD
3. In the sidebar, click **"Initialize DB (First Run)"**

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | Neon PostgreSQL connection string |
| `APP_PASSWORD` | Recommended | Password to access the app |
| `ANTHROPIC_API_KEY` | No | Anthropic Claude API key |
| `GOOGLE_API_KEY` | No | Google Gemini API key |
| `GROQ_API_KEY` | No | Groq API key |
| `OPENAI_API_KEY` | No | OpenAI API key |

---

## File Upload Limits

This application has been configured with high file upload limits:

- **Max Upload Size:** 10GB (configured in `.streamlit/config.toml`)
- **Max Message Size:** 10GB

Note: Platform-specific limits may apply:
- **Streamlit Cloud:** ~200MB limit
- **Railway/Render:** Depends on plan
- **Self-hosted:** No platform limits

For truly unlimited uploads on Streamlit Cloud, consider self-hosting.

---

## Security Considerations

1. **Always set APP_PASSWORD** for public deployments
2. **Never commit secrets** to your repository
3. Use environment variables or platform secrets management
4. Consider IP whitelisting for sensitive deployments
5. API keys entered in the UI are not stored on the server

---

## Troubleshooting

### "Connection refused" errors
- Ensure DATABASE_URL is correctly formatted
- Check Neon project is active (not suspended)
- Verify SSL mode is `require`

### Memory issues with large files
- Reduce concurrent uploads
- Use chunked processing for very large files
- Consider upgrading your hosting plan

### Session state issues
- Ensure you're using a platform that supports WebSockets
- Do NOT use Vercel, AWS Lambda, or similar serverless platforms

### PDF generation errors
- Ensure `reportlab` is installed
- Check for missing system fonts (install fonts package if needed)

---

## Local Development

For local development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8080
```

Set environment variables locally:
```bash
export DATABASE_URL="your-neon-url"
export APP_PASSWORD="local-dev-password"
```

Or create a `.streamlit/secrets.toml` file (not committed to git).
