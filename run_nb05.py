import nbformat
from nbclient import NotebookClient

with open('notebooks/05_evaluation.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='venv', resources={'metadata': {'path': 'notebooks/'}})
try:
    client.execute()
    with open('notebooks/05_evaluation.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook executed successfully.")
except Exception as e:
    print(f"Error executing notebook: {e}")
