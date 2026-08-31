import json
from pathlib import Path


def test_other_utilities_notebook_is_clean_and_compiles():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "examples" / "other_utilities.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)

    for cell in code_cells:
        compile("".join(cell["source"]), str(notebook_path), "exec")
