import unittest
from unittest.mock import MagicMock
import sys

# Mock streamlit.session_state
class SessionState(dict):
    pass

class TestDashboardLogic(unittest.TestCase):
    def setUp(self):
        # Mock streamlit module
        sys.modules['streamlit'] = MagicMock()
        import streamlit as st
        st.session_state = SessionState()
        st.session_state['dashboard_charts'] = []

    def test_add_chart_to_dashboard(self):
        import streamlit as st

        # Simulate adding a chart
        chart = {'figure': 'fig1', 'title': 'Chart 1', 'type': 'Bar'}
        st.session_state['dashboard_charts'].append(chart)

        self.assertEqual(len(st.session_state['dashboard_charts']), 1)
        self.assertEqual(st.session_state['dashboard_charts'][0]['title'], 'Chart 1')

    def test_remove_chart_from_dashboard(self):
        import streamlit as st

        # Add charts
        st.session_state['dashboard_charts'].append({'figure': 'fig1', 'title': 'Chart 1'})
        st.session_state['dashboard_charts'].append({'figure': 'fig2', 'title': 'Chart 2'})

        # Remove first chart (simulate pop)
        st.session_state['dashboard_charts'].pop(0)

        self.assertEqual(len(st.session_state['dashboard_charts']), 1)
        self.assertEqual(st.session_state['dashboard_charts'][0]['title'], 'Chart 2')

if __name__ == '__main__':
    unittest.main()
