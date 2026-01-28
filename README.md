# AI Data Analyst Application

A comprehensive AI-powered data analysis application built with Streamlit.

## Features

- **Data Ingestion**: Support for CSV, Excel, JSON, and SQL Databases (via connection string).
- **Data Profiling**: Automatic detection of data types, missing values, duplicates, and correlations.
- **Data Cleaning**: Impute missing values, remove duplicates, normalize column names.
- **Transformation**: Filter, group by, and aggregate data.
- **Analysis**: Clustering (K-Means) and other statistical analysis.
- **Visualization**: AI-suggested charts and custom chart builder using Plotly.
- **AI Insights**: Integration with Anthropic (Claude) and Google (Gemini) for natural language insights.
- **Chat Interface**: Ask questions about your data in plain English.
- **Reporting**: Export analysis reports as HTML.

## Setup Instructions

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Configuration

You can configure the application in `config/settings.yaml`.

To use AI features, you will need an API key from either Anthropic or Google. You can enter this key in the application UI.

## Project Structure

- `app.py`: Main entry point.
- `components/`: UI components for each stage of the workflow.
- `utils/`: Helper functions for data processing, loading, and LLM interaction.
- `tests/`: Unit tests.
