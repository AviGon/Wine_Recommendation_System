"""HTML export validation and output preservation."""
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

if find_spec('nbconvert') is not None:
    from scripts.export_html import export_html
else:
    export_html = None


@unittest.skipIf(export_html is None, 'Report export requires nbconvert')
class ReportExportTests(unittest.TestCase):
    def notebook(self, root, *, executed=True, error=False):
        cell = nbformat.v4.new_code_cell('1 + 1', execution_count=1 if executed else None)
        if error:
            cell.outputs = [nbformat.v4.new_output(
                'error', ename='ValueError', evalue='invalid', traceback=[])]
        elif executed:
            cell.outputs = [nbformat.v4.new_output(
                'execute_result', data={'text/plain': '2'}, execution_count=1)]
        path = root/'analysis.ipynb'
        nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
        return path

    def test_export_uses_requested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root/'nested/report.html'
            self.assertEqual(export_html(self.notebook(root), destination), destination)
            self.assertTrue(destination.read_text().startswith('<!DOCTYPE html>'))

    def test_invalid_notebooks_do_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root/'report.html'
            destination.write_text('existing report')
            for options in [{'executed': False}, {'error': True}]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    export_html(self.notebook(root, **options), destination)
                self.assertEqual(destination.read_text(), 'existing report')

    def test_failed_replace_preserves_output_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root/'report.html'
            destination.write_text('existing report')
            source = self.notebook(root)
            with patch('scripts.export_html.Path.replace', side_effect=OSError('write failed')):
                with self.assertRaises(OSError):
                    export_html(source, destination)
            self.assertEqual(destination.read_text(), 'existing report')
            self.assertEqual(set(root.iterdir()), {source, destination})


if __name__ == '__main__':
    unittest.main()
