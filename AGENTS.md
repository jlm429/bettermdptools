# bettermdptools agent guide

bettermdptools provides planning and model-free reinforcement-learning algorithms,
Gymnasium environment adapters and models, experiment entrypoints, and plotting
utilities. Transition and reward representations are core library behavior.

## Core rules

- Keep changes small, focused, and reviewable. Preserve public APIs and defaults
  unless a breaking change is explicit.
- Support behavior claims with focused evidence. For bug fixes, reproduce the
  failure through a public usage path before changing code.
- Add or update tests for behavior changes. Never delete, skip, or weaken a test to
  make a change pass.
- Treat state and action indexing, transition probabilities, rewards, terminal
  handling, and Gymnasium API semantics as correctness-critical.
- For plotting changes, keep data preparation independently testable, render to
  explicit caller-provided Matplotlib Axes, avoid process-global plotting state,
  and never close or relayout caller-owned figures.
- Make resource ownership explicit. Close environments and other resources an API
  retains on success and failure, and never close caller-owned resources.
- Change dependencies, packaging, CI, and release configuration only when the task
  requires it.
- Treat `[project]` in `pyproject.toml` as the package metadata and version owner,
  with `poetry.lock` as the resolved dependency record. Do not add parallel
  packaging metadata.
- Edit documentation sources before generated output. Never manually edit committed
  pdoc HTML.
- For substantive user-facing changes, consider whether `CHANGELOG.md` needs a
  concise `Unreleased` entry for additions, fixes, breaking changes,
  deprecations, or other externally visible behavior.
- Do not use em dashes in repository prose, comments, commit messages, or PR text.
  Do not add an agent as a commit co-author.
- Never read, print, copy, or commit secrets, credentials, `.env` files, or other
  local environment files. Do not make network calls unless the task requires them.

## Skill routing

Detailed workflows live in `.agents/skills/`:

| Task | Read |
| --- | --- |
| Locate code, tests, and ownership boundaries | [Repository orientation](.agents/skills/repository-orientation/SKILL.md) |
| Change planning or model-free algorithms | [Algorithm changes](.agents/skills/algorithm-changes/SKILL.md) |
| Change Gymnasium API integration | [Gymnasium compatibility](.agents/skills/gymnasium-compatibility/SKILL.md) |
| Change an environment model, wrapper, or discretizer | [Environment model validation](.agents/skills/environment-model-validation/SKILL.md) |
| Select and report validation commands | [Testing and validation](.agents/skills/testing-validation/SKILL.md) |
| Change docstrings, API docs, or pdoc output | [Generated documentation](.agents/skills/generated-documentation/SKILL.md) |
| Refresh the README, static README assets, or saved notebook output | [README and notebook publishing](.agents/skills/readme-notebook-publishing/SKILL.md) |

Before handoff, inspect the complete diff for unrelated changes, generated
artifacts, weakened tests, dependency drift, and local files. Report the exact
validation performed and anything skipped or unavailable.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
