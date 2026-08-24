# bettermdptools agent guide

bettermdptools provides reinforcement-learning algorithms including Q-learning,
value iteration, and policy iteration, plus experiment and visualization tooling.
It also adapts Gymnasium environments into discrete state/action spaces and the
transition and reward representations used by planning algorithms. Correct
transition and reward models are core library behavior.

## Development principles

- Keep changes small, focused, and reviewable. Avoid unrelated runtime changes.
- Add or update tests for behavior changes and regression tests for bug fixes.
- Preserve public APIs and defaults unless a breaking change is explicit.
- Support claims with behavioral evidence. Reading an implementation is not proof
  that it works.
- Validate environment-model assumptions carefully: state and action indexing,
  transition probability mass, rewards, terminal handling, and Gymnasium reset,
  step, termination, and truncation semantics.

## Repository map

| Path | Purpose |
| --- | --- |
| `bettermdptools/algorithms/` | Planning and model-free reinforcement-learning algorithms |
| `bettermdptools/envs/` | Environment models, discretization, and Gymnasium wrappers |
| `bettermdptools/experiments/` | Experiment runners, factories, types, and Optuna integration |
| `bettermdptools/utils/` | Callbacks, plotting, seeding, decorators, and policy evaluation helpers |
| `tests/` | Existing environment, algorithm integration, and plotting tests |
| `examples/` | Notebook examples for environments, algorithms, experiments, and plots |
| `pyproject.toml` | Package metadata, dependencies, Python support, and tool settings |
| `setup.py` | Legacy package installation metadata |

## Testing

- Run the most relevant existing tests for the changed behavior. Add focused
  regression coverage when fixing a bug.
- Report which tests and scenarios ran and which relevant areas remain untested.
- Never delete, skip, loosen, or rewrite a test merely to make a change pass.
- For environment or algorithm changes, exercise the behavior through its normal
  public usage path when practical, not only through internal inspection.

## Security and hygiene

- Never read, print, copy, or commit secrets, credentials, `.env` files, or other
  local environment files.
- Do not make network calls unless the task requests them.
- Use subprocesses and shell commands with explicit, narrowly scoped inputs. Avoid
  destructive commands and unsafe interpolation.
- Change dependencies deliberately and only when the task requires it.
- Do not change CI, release, or publishing configuration unless explicitly asked.
- Before handoff, inspect the complete diff for secrets, local files, generated
  artifacts, and unrelated changes.

## Skill routing

Task-specific workflows live in `.agents/skills/`:

| Task | Read |
| --- | --- |
| Change planning or model-free algorithms | [Algorithm changes](.agents/skills/algorithm-changes/SKILL.md) |
| Change an environment model, adapter, wrapper, or discretizer | [Environment model validation](.agents/skills/environment-model-validation/SKILL.md) |

Add or refine skills incrementally when a recurring repository workflow becomes
clear. Keep task-specific detail in the relevant skill instead of expanding this
always-loaded guide.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
