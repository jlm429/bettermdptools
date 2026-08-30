import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_supports_colab_python_and_numpy():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]

    assert project["requires-python"] == ">=3.12,<3.15"
    assert "numpy>=2,<3" in project["dependencies"]
    assert "numpy>=1.26,<2" not in project["dependencies"]
    assert "pygame-ce>=2.5.5,<3" in project["dependencies"]
    assert not any(
        dependency.startswith("pygame>=") for dependency in project["dependencies"]
    )
