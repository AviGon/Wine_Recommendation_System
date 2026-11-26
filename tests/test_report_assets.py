"""Report rendering supports both archived and generated experiment notebooks."""
import base64
import json
import unittest
from unittest.mock import patch
import zlib

import nbformat

from scripts.build_recommendation_report import (
    CHART_NAMES, CHART_REPORT_NAMES, chart_images, embedded_data,
)

PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII='


class ReportAssetTests(unittest.TestCase):
    def setUp(self):
        self.reports = {name: 'column\nvalue\n' for name in CHART_REPORT_NAMES}
        self.images = {name: PNG for name in CHART_NAMES}

    def published_notebook(self, reports=None):
        data = dict(reports=self.reports if reports is None else reports, images=self.images)
        archive = base64.b64encode(zlib.compress(json.dumps(data).encode())).decode()
        return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f'_ARCHIVE = {archive!r}')])

    def test_archived_analysis_figures_remain_supported(self):
        notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell(f"figure.savefig('{name}.png')", outputs=[
                nbformat.v4.new_output('display_data', data={'image/png': PNG})])
            for name in CHART_NAMES])
        with patch('scripts.build_recommendation_report.pinned', return_value=nbformat.writes(notebook).encode()):
            self.assertEqual(chart_images('revision', {'notebooks/xwines_analysis.ipynb'}, self.reports), self.images)

    def test_report_figures_work_without_detailed_notebooks(self):
        with patch('scripts.build_recommendation_report.pinned',
                   return_value=nbformat.writes(self.published_notebook()).encode()):
            self.assertEqual(chart_images('revision', {'notebooks/wine_recommendation.ipynb'}, self.reports), self.images)

    def test_stale_figures_are_rejected(self):
        changed = dict(self.reports)
        changed['dataset_summary.csv'] = 'different data\n'
        with patch('scripts.build_recommendation_report.pinned',
                   return_value=nbformat.writes(self.published_notebook(changed)).encode()):
            with self.assertRaisesRegex(ValueError, 'do not match'):
                chart_images('revision', {'notebooks/wine_recommendation.ipynb'}, self.reports)

    def test_archive_reader_does_not_execute_code(self):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "_ARCHIVE = __import__('pathlib').Path('unexpected-file').write_text('invalid')")])
        with self.assertRaises(ValueError):
            embedded_data(notebook)


if __name__ == '__main__':
    unittest.main()
