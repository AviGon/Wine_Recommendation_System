"""Export the executed cumulative notebook as a readable HTML report."""
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]


def export_workbook():
    source = ROOT/'notebooks/wine_recommendation.ipynb'
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        if cell.execution_count is None:
            raise ValueError('Execute the cumulative notebook before exporting')
        if any(output.output_type == 'error' for output in cell.outputs):
            raise ValueError('Cannot export a notebook containing execution errors')
    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_input_tags = {'report-payload'}
    exporter = HTMLExporter(config=config, template_name='lab')
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook, resources={'metadata':{'name':'X-Wines: Predicting Wine Ratings'}})
    if not body.lower().startswith('<!doctype html>'):
        raise ValueError('HTML export is missing its document type declaration')
    # The required file signature uses this exact capitalization.
    body = '<!DOCTYPE html>' + body[len('<!doctype html>'):]
    destination = ROOT/'output/workbook.html'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(body, encoding='utf-8')
    print(f'Exported {destination} ({destination.stat().st_size:,} bytes)')
    return destination


if __name__ == '__main__':
    export_workbook()
