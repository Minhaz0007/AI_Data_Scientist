import unittest
from unittest.mock import patch, MagicMock
import os
import streamlit as st
from utils.llm_helper import get_ai_response, generate_insights_prompt, get_available_provider, _clean_api_key

class TestLLMHelper(unittest.TestCase):

    def test_clean_api_key(self):
        """Test the API key cleaning utility."""
        self.assertEqual(_clean_api_key(" clean "), "clean")
        self.assertEqual(_clean_api_key('"quoted"'), "quoted")
        self.assertEqual(_clean_api_key("'quoted'"), "quoted")
        self.assertEqual(_clean_api_key(' " mixed " '), "mixed")
        self.assertEqual(_clean_api_key(""), "")
        self.assertIsNone(_clean_api_key(None))

    def test_generate_insights_prompt(self):
        summary = "Rows: 100, Cols: 5"
        results = "Mean Age: 30"
        prompt = generate_insights_prompt(summary, results)
        self.assertIn("DATA SUMMARY:", prompt)
        self.assertIn("KEY ANALYSIS RESULTS:", prompt)

    @patch('anthropic.Anthropic')
    def test_get_ai_response_anthropic(self, mock_anthropic_class):
        # Mock the client and response
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Insights generated.")]
        mock_client.messages.create.return_value = mock_message

        response = get_ai_response("prompt", "fake_key", "anthropic")
        self.assertEqual(response, "Insights generated.")

    @patch('google.genai.Client')
    def test_get_ai_response_google(self, mock_genai_client_class):
        # Mock the client instance
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.text = "Gemini insights."

        # Mock models.generate_content
        mock_client.models.generate_content.return_value = mock_response

        response = get_ai_response("prompt", "fake_key", "google")
        self.assertEqual(response, "Gemini insights.")

    def test_missing_key(self):
        response = get_ai_response("prompt", "", "anthropic")
        self.assertIn("Error: API key is missing", response)

    def test_get_available_provider_cleaning_secrets(self):
        # We simulate st.secrets using a dict
        secrets_mock = {"GROQ_API_KEY": " gsk_dirty_key "}

        with patch('streamlit.secrets', secrets_mock):
             # Also assume no other env vars are interfering
             with patch.dict(os.environ, {}, clear=True):
                 provider, key, model = get_available_provider()

                 # It should skip anthropic/openai (missing) and find groq
                 self.assertEqual(provider, 'groq')
                 self.assertEqual(key, 'gsk_dirty_key')

    def test_get_available_provider_cleaning_quotes(self):
        secrets_mock = {"GROQ_API_KEY": "\"gsk_quoted_key\""}
        with patch('streamlit.secrets', secrets_mock):
             with patch.dict(os.environ, {}, clear=True):
                 provider, key, model = get_available_provider()
                 self.assertEqual(provider, 'groq')
                 self.assertEqual(key, 'gsk_quoted_key')

if __name__ == '__main__':
    unittest.main()
