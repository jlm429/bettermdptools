---
name: testing-validation
description: Select, run, and report bettermdptools tests, formatting, lint, package builds, documentation checks, and configured no-mistakes gates. Use when validating a change or defining its verification plan.
---

# Test and validate changes

Read the repository [agent guide](../../../AGENTS.md) and the skill for the changed
subsystem first. Use the Poetry environment defined by `pyproject.toml` and
`poetry.lock`. CircleCI installs with:

```bash
poetry install --no-interaction
```

## Run narrow checks first

Select the smallest test that exercises the changed public behavior. Established
focused commands include:

```bash
poetry run python -m pytest -q tests/test_gymnasium_compat.py
poetry run python -m pytest -q tests/test_envs.py
poetry run python -m pytest -q tests/test_runtime_correctness.py
poetry run python -m pytest -q tests/test_public_api.py
poetry run python -m pytest -q tests/test_plots.py
poetry run python -m pytest -q path/to/test_file.py -k test_name
```

Use `tests/test_runtime_correctness.py` for planning and decay edge cases,
Pendulum cache failure recovery and atomic publication, rendering, and lifecycle
ownership. Use `tests/test_public_api.py` when changing documented exports. These
focused suites complement, rather than replace, the Gymnasium contract and
environment integration suites.

Then run the full local suite:

```bash
poetry run python -m pytest -q
```

CircleCI's checked-in equivalent is `poetry run pytest tests`. Do not substitute
unittest discovery, because pytest-style compatibility tests would not execute.

## Static and package checks

Run these after behavioral tests:

```bash
poetry run ruff check .
poetry run black --check .
poetry check --lock
poetry build
```

`poetry build` produces both a wheel and source distribution under ignored `dist/`.
Remove or leave ignored build output out of the commit. Documentation generation is
conditional. Follow [Generated documentation](../generated-documentation/SKILL.md)
instead of regenerating docs as a routine validation step.

When package metadata, dependency groups, or `README.md` changes, inspect both
artifacts and exercise clean wheel and source-distribution installs through a
representative public workflow. If the Optuna extra changes, validate both the base
install without Optuna and the extra-enabled path. Validate package-index-compatible
README rendering and remote assets when their presentation changes.

## Classify and report results

- Report every command and outcome, including test counts and relevant warnings.
- If a check cannot run because a tool, dependency group, platform capability, or
  external service is unavailable, report it as unavailable. Do not label it passed.
- Reproduce an unexpected failure on the unchanged base revision, or equivalent
  clean-base environment, before calling it pre-existing. Otherwise treat it as a
  possible regression.
- Do not hide a failure by weakening tests or excluding paths. If an established
  failure is outside the authorized scope, preserve the evidence and report the
  boundary.
- State what was not exercised, especially slow model generation, stochastic paths,
  rendering, optional dependencies, and generated docs.

## no-mistakes gate

The no-mistakes repository integration is external to the tracked tree, and this
repository does not commit a no-mistakes test command. When the user asks to gate or
ship changes, first confirm initialization with `no-mistakes axi`, then follow the
loaded `$no-mistakes` workflow after committing on a feature branch. Report the
pipeline's review, test, documentation, lint, build, PR, and CI outcomes. Do not
treat an agent-driven gate as a substitute for listing the concrete commands it ran.
