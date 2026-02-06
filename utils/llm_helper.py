import os
import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# LLM Provider configurations with environment variable names
LLM_PROVIDERS = {
    'anthropic': {
        'name': 'Anthropic (Claude)',
        'env_var': 'ANTHROPIC_API_KEY',
        'models': [
            'claude-sonnet-4-20250514',
            'claude-3-5-sonnet-20241022',
            'claude-3-opus-20240229',
            'claude-3-haiku-20240307'
        ],
        'default_model': 'claude-sonnet-4-20250514'
    },
    'openai': {
        'name': 'OpenAI (GPT)',
        'env_var': 'OPENAI_API_KEY',
        'models': [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4-turbo',
            'gpt-3.5-turbo'
        ],
        'default_model': 'gpt-4o'
    },
    'groq': {
        'name': 'Groq',
        'env_var': 'GROQ_API_KEY',
        'models': [
            'llama-3.3-70b-versatile',
            'llama-3.1-8b-instant',
            'mixtral-8x7b-32768',
            'gemma2-9b-it'
        ],
        'default_model': 'llama-3.3-70b-versatile'
    },
    'mistral': {
        'name': 'Mistral AI',
        'env_var': 'MISTRAL_API_KEY',
        'models': [
            'mistral-large-latest',
            'mistral-medium-latest',
            'mistral-small-latest',
            'open-mixtral-8x22b'
        ],
        'default_model': 'mistral-large-latest'
    },
    'together': {
        'name': 'Together AI',
        'env_var': 'TOGETHER_API_KEY',
        'models': [
            'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
            'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'Qwen/Qwen2.5-72B-Instruct-Turbo'
        ],
        'default_model': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    },
    'cohere': {
        'name': 'Cohere',
        'env_var': 'COHERE_API_KEY',
        'models': [
            'command-r-plus',
            'command-r',
            'command-light'
        ],
        'default_model': 'command-r-plus'
    },
    'google': {
        'name': 'Google (Gemini)',
        'env_var': 'GOOGLE_API_KEY',
        'models': [
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-1.5-pro',
            'gemini-1.5-flash'
        ],
        'default_model': 'gemini-2.0-flash'
    }
}

# Provider priority order for auto-detection
PROVIDER_PRIORITY = ['anthropic', 'openai', 'groq', 'mistral', 'together', 'cohere', 'google']


def _clean_api_key(key):
    """Clean API key by removing whitespace and quotes."""
    if isinstance(key, str):
        return key.strip().strip('"').strip("'").strip()
    return key


def get_available_provider():
    """
    Auto-detect available LLM provider based on configured API keys.
    Checks environment variables and Streamlit secrets.

    Returns:
        tuple: (provider_code, api_key, model) or (None, None, None) if no provider available
    """
    for provider in PROVIDER_PRIORITY:
        config = LLM_PROVIDERS[provider]
        env_var = config['env_var']

        # Check Streamlit secrets first, then environment variables
        api_key = None
        try:
            if hasattr(st, 'secrets') and env_var in st.secrets:
                api_key = st.secrets[env_var]
        except Exception:
            pass

        if not api_key:
            api_key = os.environ.get(env_var)

        if api_key:
            return provider, _clean_api_key(api_key), config['default_model']

    return None, None, None


def get_all_available_providers():
    """
    Get all available LLM providers based on configured API keys.

    Returns:
        list: List of tuples (provider_code, api_key, default_model) for available providers
    """
    available = []
    for provider in PROVIDER_PRIORITY:
        config = LLM_PROVIDERS[provider]
        env_var = config['env_var']

        # Check Streamlit secrets first, then environment variables
        api_key = None
        try:
            if hasattr(st, 'secrets') and env_var in st.secrets:
                api_key = st.secrets[env_var]
        except Exception:
            pass

        if not api_key:
            api_key = os.environ.get(env_var)

        if api_key:
            available.append((provider, _clean_api_key(api_key), config['default_model']))

    return available

@st.cache_resource
def _get_cached_client(provider, api_key):
    """
    Internal function to get or create an LLM client, cached by Streamlit.
    This is used for providers where the client object encapsulates the connection/state.
    """
    if provider == 'anthropic':
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    elif provider == 'openai':
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    elif provider == 'groq':
        from groq import Groq
        return Groq(api_key=api_key)
    elif provider == 'mistral':
        from mistralai import Mistral
        return Mistral(api_key=api_key)
    elif provider == 'together':
        from together import Together
        return Together(api_key=api_key)
    elif provider == 'cohere':
        import cohere
        return cohere.ClientV2(api_key=api_key)
    else:
        return None

def get_llm_client(provider, api_key):
    """
    Get an LLM client. Uses caching for supported providers.
    Google provider is not cached due to global state configuration.
    """
    if provider == 'google':
        from google import genai
        return genai.Client(api_key=api_key)
    else:
        return _get_cached_client(provider, api_key)

# Groq model fallback order: try smaller/cheaper models when rate-limited
GROQ_FALLBACK_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'gemma2-9b-it',
]

# Maximum retries for transient errors (not rate limits)
_GROQ_MAX_RETRIES = 2
_GROQ_RETRY_DELAY = 1  # seconds


def _is_rate_limit_error(exc):
    """Check if an exception is a rate limit (429) error."""
    # Groq SDK raises groq.RateLimitError for 429 responses
    exc_type = type(exc).__name__
    if exc_type == 'RateLimitError':
        return True
    # Fallback: check the string representation
    err_str = str(exc)
    if '429' in err_str or 'rate_limit' in err_str.lower() or 'rate limit' in err_str.lower():
        return True
    return False


def _groq_request_with_fallback(client, model_name, prompt, max_tokens):
    """Make a Groq API request with automatic fallback to smaller models on rate limit."""
    # Build fallback chain starting from the requested model
    models_to_try = [model_name]
    for fallback in GROQ_FALLBACK_MODELS:
        if fallback != model_name and fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = None
    for current_model in models_to_try:
        for attempt in range(_GROQ_MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if _is_rate_limit_error(e):
                    logger.warning(
                        "Groq rate limit hit for model '%s' (attempt %d). %s",
                        current_model, attempt + 1, str(e)
                    )
                    # Don't retry same model on rate limit — move to fallback
                    break
                else:
                    # Transient error — retry with backoff
                    if attempt < _GROQ_MAX_RETRIES:
                        time.sleep(_GROQ_RETRY_DELAY * (attempt + 1))
                    else:
                        raise

    # All models exhausted — return a user-friendly message
    return (
        "**Rate limit reached** — Your Groq free-tier daily token quota has been exceeded "
        "for all available models.\n\n"
        "**What you can do:**\n"
        "- Wait ~15-30 minutes for the limit to partially reset, then try again\n"
        "- Upgrade to Groq's Dev Tier at https://console.groq.com/settings/billing "
        "for higher limits\n"
        "- Configure a different AI provider (OpenAI, Anthropic, Google) as a backup"
    )


def get_ai_response(prompt, api_key, provider='anthropic', model=None, max_tokens=4096):
    """
    Generate a response from an LLM.

    Args:
        prompt: The prompt text.
        api_key: The API key.
        provider: Provider code (anthropic, openai, groq, mistral, together, cohere, google).
        model: Specific model to use (optional, uses provider default if not specified).
        max_tokens: Maximum tokens in the response.
    """
    # Fallback to environment variables if api_key is not provided
    if not api_key and provider in LLM_PROVIDERS:
        env_var = LLM_PROVIDERS[provider]['env_var']
        # Check Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and env_var in st.secrets:
                api_key = st.secrets[env_var]
        except Exception:
            pass
        # Then check environment variables
        if not api_key:
            api_key = os.environ.get(env_var)

    if not api_key:
        return "Error: API key is missing. Please configure it in Streamlit secrets or environment variables."

    api_key = _clean_api_key(api_key)

    try:
        client = get_llm_client(provider, api_key)

        if provider == 'anthropic':
            model_name = model or LLM_PROVIDERS['anthropic']['default_model']
            message = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text

        elif provider == 'openai':
            model_name = model or LLM_PROVIDERS['openai']['default_model']
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif provider == 'groq':
            model_name = model or LLM_PROVIDERS['groq']['default_model']
            return _groq_request_with_fallback(client, model_name, prompt, max_tokens)

        elif provider == 'mistral':
            model_name = model or LLM_PROVIDERS['mistral']['default_model']
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif provider == 'together':
            model_name = model or LLM_PROVIDERS['together']['default_model']
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif provider == 'cohere':
            model_name = model or LLM_PROVIDERS['cohere']['default_model']
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.message.content[0].text

        elif provider == 'google':
            model_name = model or LLM_PROVIDERS['google']['default_model']
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text

        else:
            return f"Error: Unsupported provider '{provider}'."

    except Exception as e:
        if _is_rate_limit_error(e):
            return (
                "**Rate limit reached** — Your API token quota has been exceeded.\n\n"
                "Please wait a few minutes and try again, or upgrade your plan "
                "for higher limits."
            )
        return f"Error generating response: {str(e)}"

def generate_insights_prompt(df_summary, analysis_results):
    """
    Constructs a prompt for generating insights.
    """
    prompt = f"""
    You are an expert data analyst. Based on the following analysis of a dataset:

    DATA SUMMARY:
    {df_summary}

    KEY ANALYSIS RESULTS:
    {analysis_results}

    Please provide:
    1. Top 5 key insights from the data.
    2. Business implications of these findings.
    3. Actionable recommendations.
    4. Potential concerns or limitations in the data.

    Format the output clearly with headings.
    """
    return prompt

def generate_chat_prompt(data_context, user_question, chat_history=None):
    """
    Constructs a prompt for the chat interface.
    """
    history_context = ""
    if chat_history:
        history_context = "\n\nPrevious conversation:\n"
        for msg in chat_history[-5:]:  # Last 5 messages for context
            history_context += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""
    You are a data assistant. Here is the context of the dataset:
    {data_context}
    {history_context}

    User Question: {user_question}

    Answer the question based on the data provided. If you need to perform complex analysis, describe the steps.
    If asked to show code, provide Python pandas code snippets.
    """
    return prompt
