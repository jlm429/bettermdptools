# Experiments Entrypoint API

The experiments package provides a high-level entry point for running one
environment and algorithm experiment with a consistent return shape.

This layer is intended for quick iteration, demos, and light experimentation.
It is not required when using bettermdptools algorithms directly.

Primary entry point:
- `bettermdptools.experiments.run(...)`

## Typical workflow

A complete run usually follows this pattern:

1. Choose an environment ID.
2. Choose an algorithm name.
3. Optionally provide algorithm and environment keyword arguments.
4. Optionally evaluate the learned policy.

Example:

```python
from bettermdptools.experiments import run

out = run(
    algo="vi",
    env_id="FrozenLake-v1",
    seed=0,
    env_kwargs={"is_slippery": False},
    algo_kwargs={"gamma": 0.99, "n_iters": 2000, "theta": 1e-10},
    eval_kwargs={"n_iters": 200, "render": False},
)

print(out.algo, out.env_id, out.seed)
print(out.train.keys())
print(out.eval["scores"][:5])
```

## Responsibilities

### What this layer handles

- Environment creation via `gymnasium.make`
- Obtaining a Gymnasium-style transition dictionary `P` as described in
  [Notes on wrappers and `P`](#notes-on-wrappers-and-p)
  - Applies a wrapper when a usable native tabular model is unavailable
- Dispatching to Planner algorithms (`vi`, `pi`) or tabular RL algorithms (`q_learning`, `sarsa`)
- Returning a consistent `RunResult` object
- Optional evaluation via `TestEnv.test_env` when `eval_kwargs` is provided

## Public entrypoint

### `run(...)`

```python
from bettermdptools.experiments import run
out = run(algo="q_learning", env_id="Taxi-v4")
```

#### Parameters

- `algo: str`  
  Supported names include "vi", "pi", "q_learning" (or "q"), and "sarsa".

- `env_id: str`  
  Gymnasium environment id string.

- `seed: Optional[int]`  
  Seeds global random number generators. It is also the default for the first
  model-free training reset and the first evaluation reset. Explicit seeds in
  `algo_kwargs` or `eval_kwargs` take precedence.

- `env_kwargs: Optional[Dict[str, Any]]`  
  Forwarded to `gymnasium.make`.

- `wrapper: Optional[Callable | str]`  
  Optional environment wrapper applied when a usable native tabular model is
  not exposed.

- `wrapper_kwargs: Optional[Dict[str, Any]]`  
  Forwarded to the wrapper constructor.

- `algo_kwargs: Optional[Dict[str, Any]]`  
  Forwarded to the selected algorithm call.

- `eval_kwargs: Optional[Dict[str, Any]]`  
  If nonempty, evaluates the learned policy. Evaluation does not require a
  rendering backend when `render` is omitted or false. For local or CI
  rendering, install `bettermdptools[rendering]`, which uses `pygame-ce`
  through its compatible `pygame` import namespace. Classic `pygame` is not
  supported.

> **Important:** Rendering is not supported on Google Colab. Non-rendering
> planning, training, evaluation, plotting, and experiment workflows work
> normally there.

#### Returns

`RunResult`, exported from `bettermdptools.experiments`, containing:

- `algo`
- `env_id`
- `seed`
- `train`
- `eval`
- `meta`

Common training keys:

- Planner algorithms (`vi`, `pi`)
  - `V`, `V_track`, `pi`
  - `planning_metadata` when `algo_kwargs` requests `return_metadata=True`

- Tabular RL algorithms (`q_learning`, `sarsa`)
  - `Q`, `V`, `pi`, `Q_track`, `pi_track`, `rewards`

## Notes on wrappers and `P`

Many planning and tabular RL methods require a Gymnasium-style transition
dictionary `P` and discrete state and action spaces.

- Built-in discrete models are read from `env.unwrapped.P` when both spaces are
  discrete.
- Explicit wrapper models are read from the wrapper's `env.P` property.
- Otherwise, a wrapper can be provided to adapt the environment.
- The internal registry includes Blackjack, CartPole, Acrobot, and Pendulum.

If `P` cannot be obtained, `run(...)` raises an error.

## Stability notes

This entrypoint is intended to be helpful and lightweight.

The most stable contract is:

- `run(...)` returns a `RunResult`
- `RunResult.train` and `RunResult.eval` are dictionaries
- The presence of `pi` in `train` is required for evaluation

## Examples

[`../../examples/experiments_demo.ipynb`](../../examples/experiments_demo.ipynb)
