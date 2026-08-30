import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_supports_colab_python_and_numpy():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]

    assert project["requires-python"] == ">=3.12,<3.15"
    assert "numpy>=2,<3" in project["dependencies"]
    assert "numpy>=1.26,<2" not in project["dependencies"]

    rendering_requirements = [
        Requirement(dependency)
        for dependency in project["dependencies"]
        if Requirement(dependency).name in {"pygame", "pygame-ce"}
    ]

    def selected_renderers(python_version):
        return {
            requirement.name
            for requirement in rendering_requirements
            if requirement.marker.evaluate({"python_version": python_version})
        }

    assert selected_renderers("3.11") == set()
    assert selected_renderers("3.12") == {"pygame"}
    assert selected_renderers("3.13") == {"pygame"}
    assert selected_renderers("3.14") == {"pygame-ce"}
    assert selected_renderers("3.15") == set()
