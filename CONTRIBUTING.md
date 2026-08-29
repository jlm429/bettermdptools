# Contributing to bettermdptools

Contributions should be small, focused, and supported by tests that exercise the
public behavior being changed. Transition probabilities, rewards, terminal
handling, state and action indexing, and Gymnasium API semantics are
correctness-critical.

## Set up the Poetry environment

Use a Python version allowed by `[project].requires-python` in `pyproject.toml` and
install [Poetry](https://python-poetry.org/docs/#installation), then run:

```bash
poetry install --with docs
```

To include the optional Optuna integration:

```bash
poetry install --with docs --extras optuna
```

Use `poetry run` for repository commands. Do not rely on packages installed in
an unrelated Python environment.

## Make and validate a change

Create a feature branch and add or update focused tests with the implementation.
Run the narrowest relevant test first, then the complete local checks:

```bash
poetry run pytest -q
poetry run ruff check .
poetry run black --check .
```

Ruff can apply safe fixes with `poetry run ruff check . --fix`. Format Python
sources with `poetry run black .`, then rerun the checks above.

## Update API documentation

Python docstrings are the source for the generated API reference. When a
docstring or public Python API changes, regenerate the committed pdoc output:

```bash
poetry install --with docs
poetry run pdoc bettermdptools -o docs
```

Do not manually edit generated HTML or `docs/search.js`. Preserve the
hand-authored `docs/index.html` redirect and Markdown files under `docs/api/`.
Run the generation command a second time and confirm that it produces no
additional diff.

`README.md` is also the package long description. When changing its rendering or
assets, check both repository and package-index-compatible rendering. Repository
images must use stable absolute URLs in the package long description.

## Build and submit

Before opening a pull request, validate the package metadata and artifacts:

```bash
poetry check --lock
poetry build
```

Inspect the complete diff for unrelated files, generated notebook output,
dependency drift, and weakened tests. Commit only the intended work, push the
feature branch, and open a pull request that explains the behavior and lists
the exact validation performed.

When package metadata, dependency groups, or the package long description changes,
inspect both built artifacts and exercise clean wheel and source-distribution
installs through a representative public workflow. Keep `dist/` and other build or
installation scratch files out of the commit.
