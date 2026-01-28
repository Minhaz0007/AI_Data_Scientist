import os
import anthropic
import google.generativeai as genai

def get_ai_response(prompt, api_key, provider='anthropic'):
    """
    Generate a response from an LLM.

    Args:
        prompt: The prompt text.
        api_key: The API key.
        provider: 'anthropic' or 'google'.
    """
    if not api_key:
        return "Error: API key is missing."

    try:
        if provider == 'anthropic':
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text

        elif provider == 'google':
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text

        else:
            return "Error: Unsupported provider."

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
