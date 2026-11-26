"""Export an executed notebook as a standalone HTML report."""
from pathlib import Path
import argparse
import tempfile
import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]


def export_html(source=None, destination=None):
    source = Path(source) if source is not None else ROOT/'notebooks/wine_recommendation.ipynb'
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        if cell.execution_count is None:
            raise ValueError('Notebook contains an unexecuted code cell')
        if any(output.output_type == 'error' for output in cell.outputs):
            raise ValueError('Cannot export a notebook containing execution errors')
    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_input_tags = {'report-payload'}
    exporter = HTMLExporter(config=config, template_name='lab')
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    title = notebook.metadata.get('title', source.stem.replace('_', ' ').title())
    body, _ = exporter.from_notebook_node(notebook, resources={'metadata': {'name': title}})
    if not body.lower().startswith('<!doctype html>'):
        raise ValueError('HTML export is missing its document type declaration')
    body = '<!DOCTYPE html>' + body[len('<!doctype html>'):]
    destination = Path(destination) if destination is not None else ROOT/'output/workbook.html'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         suffix='.html', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(body)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f'Exported {destination} ({destination.stat().st_size:,} bytes)')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', type=Path, help='Executed source notebook')
    parser.add_argument('--output', type=Path, help='Destination HTML file')
    arguments = parser.parse_args()
    export_html(arguments.notebook, arguments.output)
