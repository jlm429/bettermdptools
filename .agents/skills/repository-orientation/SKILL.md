---
name: repository-orientation
description: Locate bettermdptools architecture, public integration paths, tests, documentation ownership, and authoritative configuration before making a change. Use when scoping unfamiliar or cross-cutting repository work.
---

# Orient within bettermdptools

Read the repository [agent guide](../../../AGENTS.md) first. Use
[Testing and validation](../testing-validation/SKILL.md) for commands and reporting.

## Runtime architecture

- `bettermdptools/algorithms/planner.py` consumes tabular `P` dictionaries for value
  iteration and policy iteration. `algorithms/rl.py` interacts directly with
  Gymnasium environments for Q-learning and SARSA.
- `bettermdptools/envs/` owns custom transition models, discretization, observation
  and action transforms, and Gymnasium wrappers.
- `bettermdptools/experiments/env_factory.py` creates environments, obtains built-in
  models from `env.unwrapped.P`, applies explicit or registered wrappers, and returns
  `EnvBundle`. `experiments/run.py` dispatches training and optional evaluation.
- `bettermdptools/utils/test_env.py` evaluates policies through environment steps.
  `utils/seed.py` supplies best-effort global seeding. Other utilities own callbacks,
  decorators, and plots.

Trace changes through these public integration points. Do not infer environment
support from a filename or notebook. Inspect the current factory registry, wrapper,
tests, and construction cost before making a support claim.

## Evidence and examples

- `tests/test_gymnasium_compat.py` covers current reset and step contracts, wrapper
  spaces, truncation, model access, model invariants, and TD boundaries.
- `tests/test_envs.py` provides algorithm and wrapper integration smoke coverage.
- `tests/test_plots.py` covers plotting utilities.
- `examples/*.ipynb` are user-facing workflows, but saved notebook output is not a
  substitute for a test.

## Configuration and documentation

- `pyproject.toml` and `poetry.lock` define the Poetry environment, supported Python
  and Gymnasium ranges, dev tools, formatting, lint, and build backend.
- `setup.py` repeats legacy package metadata and runtime requirements. Inspect both
  metadata surfaces when a packaging change is explicitly required.
- `.circleci/config.yml` is the checked-in CI workflow.
- `README.md` and `docs/api/*.md` are hand-authored docs. Python docstrings and
  `docs-templates/` feed committed pdoc output under `docs/`. Read
  [Generated documentation](../generated-documentation/SKILL.md) before changing
  either side.

Use `rg --files` and `rg` to confirm paths and call sites. Git history is evidence
for intentional conventions, especially when generated files or duplicated metadata
make ownership unclear. Remove stale or speculative guidance instead of documenting
an assumption as repository behavior.
