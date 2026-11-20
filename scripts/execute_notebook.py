"""Execute all cells using this Python interpreter and fail on the first error."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
(ROOT/'.cache').mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.cache/ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.cache/jupyter-runtime'))
path = ROOT/'notebooks/person_a_tasks_1_2.ipynb'
nb = nbformat.read(path,as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client = NotebookClient(nb,km=km,timeout=1200,resources={'metadata':{'path':str(ROOT)}})
client.on_cell_start = lambda cell, cell_index: print(f'Cell {cell_index+1}/{len(nb.cells)}: {cell.cell_type}',flush=True)
client.execute()
nbformat.validate(nb)
nbformat.write(nb,path)
print(f'Executed and validated {path}: {sum(c.cell_type=="code" for c in nb.cells)} code cells')
