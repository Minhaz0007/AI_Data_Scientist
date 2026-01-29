import unittest
import pandas as pd
from utils.report_generator import generate_html_report, generate_pdf_report, generate_docx_report

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        self.insights = "Some dummy insights."

    def test_generate_html_report(self):
        report = generate_html_report(self.df, self.insights)
        self.assertIsInstance(report, str)
        self.assertIn("<html>", report)
        self.assertIn("Some dummy insights.", report)

    def test_generate_pdf_report(self):
        report = generate_pdf_report(self.df, self.insights)
        self.assertIsInstance(report, bytes)
        self.assertTrue(len(report) > 0)
        self.assertTrue(report.startswith(b'%PDF'))

    def test_generate_docx_report(self):
        report = generate_docx_report(self.df, self.insights)
        self.assertIsInstance(report, bytes)
        self.assertTrue(len(report) > 0)
        # DOCX is a zip file, starts with PK
        self.assertTrue(report.startswith(b'PK'))

if __name__ == '__main__':
    unittest.main()
