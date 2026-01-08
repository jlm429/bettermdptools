# Experiments Entrypoint API

The experiments package provides an optional, high-level entry point for running a single environment and algorithm experiment with a consistent return shape.

This layer is provided on an "as is" basis. It is intended for quick iteration, demos, and light experimentation. It is not required for using bettermdptools algorithms directly.

Primary entry point:
- `bettermdptools.experiments.run(...)`

## Typical workflow

A complete run usually follows this pattern:

1) Choose an environment id  
2) Choose an algorithm name  
3) Optionally provide algorithm and environment kwargs  
4) Optionally evaluate the learned policy  

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

- Environment creation via `gym.make`
- Obtaining a Gym-style transition dictionary `P`
  - Uses `env.P` or `env.unwrapped.P` when present
  - Applies a wrapper when needed (explicit or registry-based)
- Dispatching to Planner algorithms (`vi`, `pi`) or tabular RL algorithms (`q_learning`, `sarsa`)
- Returning a consistent `RunResult` object
- Optional evaluation via `TestEnv.test_env` when `eval_kwargs` is provided

## Public entrypoint

### `run(...)`

```python
from bettermdptools.experiments import run
out = run(algo="q_learning", env_id="Taxi-v3")
```

#### Parameters

- `algo: str`  
  Supported names include "vi", "pi", "q_learning" (or "q"), and "sarsa".

- `env_id: str`  
  Gymnasium environment id string.

- `seed: Optional[int]`  
  Best-effort global seeding.

- `env_kwargs: Optional[Dict[str, Any]]`  
  Forwarded to `gym.make`.

- `wrapper: Optional[Callable | str]`  
  Optional environment wrapper applied when `P` is not exposed.

- `wrapper_kwargs: Optional[Dict[str, Any]]`  
  Forwarded to the wrapper constructor.

- `algo_kwargs: Optional[Dict[str, Any]]`  
  Forwarded to the selected algorithm call.

- `eval_kwargs: Optional[Dict[str, Any]]`  
  If provided, evaluates the learned policy.

#### Returns

`RunResult` containing:

- `algo`
- `env_id`
- `seed`
- `train`
- `eval`
- `meta`

Common training keys:

- Planner algorithms (`vi`, `pi`)
  - `V`, `V_track`, `pi`

- Tabular RL algorithms (`q_learning`, `sarsa`)
  - `Q`, `V`, `pi`, `Q_track`, `pi_track`, `rewards`

## Notes on wrappers and `P`

Many planning and tabular RL methods require a Gym-style transition dictionary `P` and discrete state and action spaces.

- If `env.P` (or `env.unwrapped.P`) exists, it is used directly.
- Otherwise, a wrapper can be provided to adapt the environment.
- Some environments may be supported by a small internal wrapper registry.

If `P` cannot be obtained, `run(...)` raises an error.

## Stability notes

This entrypoint is intended to be helpful and lightweight.

The most stable contract is:

- `run(...)` returns a `RunResult`
- `RunResult.train` and `RunResult.eval` are dictionaries
- The presence of `pi` in `train` is required for evaluation

## Examples

[`../../examples/experiments_demo.ipynb`](../../examples/experiments_demo.ipynb) 
