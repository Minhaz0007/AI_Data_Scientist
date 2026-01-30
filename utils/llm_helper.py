import os
import streamlit as st

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
            return provider, api_key, config['default_model']

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
            available.append((provider, api_key, config['default_model']))

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
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

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
