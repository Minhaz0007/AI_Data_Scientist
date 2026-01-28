# AI Data Analyst Application

A comprehensive AI-powered data analysis application built with Streamlit. Upload your data, get automated insights, and generate professional reports.

## Features

### Data Ingestion
- **Multi-format support:** CSV, Excel (.xlsx, .xls, .xlsm), JSON
- **SQL Database connection:** Connect to any SQL database via connection string
- **No file size limits:** Configured for large file uploads
- **Drag-and-drop interface**

### Data Profiling
- Automatic data type detection
- Missing value analysis with visualization
- Duplicate row detection
- Correlation matrix heatmap
- Statistical summaries (mean, median, std, skewness, kurtosis)
- Column distribution visualization

### Data Cleaning
- Multiple imputation strategies (mean, median, mode, constant, drop)
- Duplicate removal with confirmation
- Column name standardization (snake_case)
- Before/after comparison

### Data Transformation
- **Filtering:** Multiple conditions (equals, contains, between, etc.)
- **Aggregation:** Group by with multiple aggregation methods
- **Pivot/Unpivot:** Reshape data easily
- **Calculated Columns:** Create new columns with Python expressions
- **Merge Datasets:** Join multiple datasets with various join types

### Analysis Engine
- **K-Means Clustering:** Segment your data with visualization
- **Outlier Detection:** IQR and Z-Score methods with treatment options
- **Statistical Tests:**
  - Normality test (Shapiro-Wilk)
  - T-test (compare two columns)
  - Pearson correlation test
  - Chi-square test for independence

### Visualization
- **AI-suggested charts** based on data patterns
- **Custom chart builder** with 6 chart types:
  - Scatter plots (with color encoding)
  - Line plots
  - Bar charts (with aggregation)
  - Histograms
  - Box plots
  - Heatmaps
- Interactive Plotly visualizations

### AI-Powered Insights
- **Multiple LLM providers:**
  - Anthropic Claude (Claude 3.5 Sonnet, Opus, Haiku)
  - Google Gemini (1.5 Pro, 1.5 Flash)
  - OpenAI GPT (GPT-4o, GPT-4 Turbo)
- **Model selection** for each provider
- **Configurable token limits**
- Comprehensive data analysis with actionable recommendations

### Chat Interface
- Natural language queries about your data
- Conversation history
- Example prompts to get started
- Multi-provider support

### Report Generation
- **Multiple formats:**
  - HTML (professional styling, print-ready)
  - PDF (via ReportLab)
  - Word/DOCX (via python-docx)
- **Data export:** CSV, Excel, JSON
- Includes AI insights when available
- Executive summary with key metrics

### Project Management
- Save/load analysis projects to Neon PostgreSQL
- Session state management
- Password protection for deployed apps

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-data-analyst.git
cd ai-data-analyst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | Neon PostgreSQL connection string |
| `APP_PASSWORD` | Recommended | Password to access the app |
| `ANTHROPIC_API_KEY` | No | Anthropic Claude API key |
| `GOOGLE_API_KEY` | No | Google Gemini API key |
| `OPENAI_API_KEY` | No | OpenAI API key |

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Recommended platforms:**
- Streamlit Cloud (free, easiest)
- Railway (Docker-based)
- Render (Docker-based)

**Note:** Vercel is NOT compatible with Streamlit applications due to WebSocket requirements.

## Project Structure

```
ai-data-analyst/
├── app.py                    # Main application entry point
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── config/
│   └── settings.yaml        # Application configuration
├── components/              # UI components
│   ├── ingestion.py        # Data upload
│   ├── profiling.py        # Data profiling
│   ├── cleaning.py         # Data cleaning
│   ├── transformation.py   # Data transformation
│   ├── analysis.py         # Statistical analysis
│   ├── visualization.py    # Chart generation
│   ├── insights.py         # AI insights
│   ├── chat.py            # Chat interface
│   └── reporting.py       # Report generation
├── utils/                  # Helper modules
│   ├── data_loader.py     # File loading
│   ├── data_processor.py  # Data operations
│   ├── llm_helper.py      # LLM integration
│   ├── report_generator.py # Report generation
│   └── db.py              # Database operations
├── tests/                  # Unit tests
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **LLM Integration:** Anthropic, Google GenAI, OpenAI
- **Database:** PostgreSQL (Neon)
- **Report Generation:** ReportLab (PDF), python-docx (Word)

## API Key Setup

You can enter API keys in two ways:

1. **In the UI:** Enter your API key in the AI Insights or Chat sections
2. **Environment Variables:** Set `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`

Get your API keys:
- [Anthropic Console](https://console.anthropic.com/)
- [Google AI Studio](https://aistudio.google.com/)
- [OpenAI Platform](https://platform.openai.com/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
