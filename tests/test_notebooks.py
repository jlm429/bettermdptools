import json
from pathlib import Path


def test_other_utilities_notebook_is_clean_and_uses_public_utility_apis():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "examples" / "other_utilities.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    assert "callbacks=metrics" in source
    assert "context: TransitionContext" in source
    assert "def learned_policy(state, info)" in source
    assert "show=False, ax=value_ax" in source
    assert "show=False,\n    ax=policy_ax" in source

    for cell in code_cells:
        compile("".join(cell["source"]), str(notebook_path), "exec")
