import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_supports_colab_python_and_numpy():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]

    supported_python = SpecifierSet(project["requires-python"])
    assert Version("3.11") not in supported_python
    assert Version("3.12") in supported_python
    assert Version("3.13") in supported_python
    assert Version("3.14") in supported_python
    assert Version("3.15") not in supported_python

    requirements = [Requirement(dependency) for dependency in project["dependencies"]]
    numpy_requirements = [
        requirement for requirement in requirements if requirement.name == "numpy"
    ]
    assert len(numpy_requirements) == 1
    supported_numpy = numpy_requirements[0].specifier
    assert Version("1.26.4") not in supported_numpy
    assert Version("2.0.0") in supported_numpy
    assert Version("2.0.2") in supported_numpy
    assert Version("2.999") in supported_numpy
    assert Version("3.0.0") not in supported_numpy

    rendering_requirements = [
        requirement
        for requirement in requirements
        if requirement.name in {"pygame", "pygame-ce"}
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
