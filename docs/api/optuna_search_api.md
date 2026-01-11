# Optuna Search Entrypoint API

The experiments package provides an optional, high-level entry point for running Optuna-based hyperparameter search on top of `bettermdptools.experiments.run(...)`, returning a consistent result shape.

This layer is provided on an **"as is" basis**. It is intended for quick tuning, demos, and light experimentation. It is not required for using bettermdptools algorithms directly, nor is it required for using Optuna directly.

Optuna is an **optional dependency**. If Optuna is not installed, calling the entrypoint raises `MissingOptunaDependency`.

---

## Primary entry point

```
bettermdptools.experiments.optimize(...)
```

---

## Typical workflow

A complete search usually follows this pattern:

1. Choose an environment id
2. Choose an algorithm name
3. Define a `suggest(trial)` function that proposes hyperparameters
4. Define a `metric(out)` function that scores a `RunResult`
5. Optionally pass fixed “base” kwargs applied to every trial
6. Run a study for `n_trials` and inspect the best run

---

## Responsibilities

### What this layer handles

- Lazy importing Optuna (optional dependency)
- Creating an Optuna `Study` (TPE sampler by default)
- Running an objective where each trial calls `experiments.run(...)`
- Merging base kwargs with per-trial suggested kwargs
- Tracking and returning a consistent “best result” bundle

---

## Public entrypoint

### optimize(...)

```python
from bettermdptools.experiments import optimize
res = optimize(algo="sarsa", env_id="Blackjack-v1", suggest=suggest, metric=metric)
```

---

## Parameters

### algo : str

Algorithm name. Supported values include:

- `"vi"`
- `"pi"`
- `"q_learning"` (or `"q"`)
- `"sarsa"`

---

### env_id : str

Gymnasium environment id string.

---

### metric : Callable[[RunResult], float]

Function mapping the `RunResult` returned by `experiments.run(...)` to a scalar objective value.

Higher values are better when `direction="maximize"`.

---

### suggest : Callable[[optuna.trial.Trial], Dict[str, Dict[str, Any]]]

Function mapping an Optuna trial to a dictionary of keyword-argument blocks.

Expected structure:

```python
{
  "env_kwargs": {...},
  "wrapper_kwargs": {...},
  "algo_kwargs": {...},
  "eval_kwargs": {...},
}
```

Any block may be omitted. Per-trial values override base kwargs.

---

### n_trials : int (default: 50)

Number of Optuna trials to run.

---

### seed : Optional[int]

Best-effort seed applied to `experiments.run(...)` and used to seed the Optuna sampler.

---

### base_env_kwargs : Optional[Dict[str, Any]]

Keyword arguments forwarded to `gym.make(env_id, **base_env_kwargs)` for all trials.

---

### base_wrapper_kwargs : Optional[Dict[str, Any]]

Keyword arguments forwarded to the wrapper constructor for all trials.

---

### base_algo_kwargs : Optional[Dict[str, Any]]

Keyword arguments forwarded to the algorithm call for all trials.

---

### base_eval_kwargs : Optional[Dict[str, Any]]

Keyword arguments forwarded to policy evaluation for all trials.

---

### direction : str (default: "maximize")

One of `"maximize"` or `"minimize"`.

---

### study_name : Optional[str]

Name of the Optuna study. Used when resuming or sharing a persistent study.

---

### storage : Optional[str]

Optuna storage URL (for example: `sqlite:///optuna.db`). Enables persistent and shared studies.

---

### load_if_exists : bool (default: False)

If `True`, load an existing study from `storage` instead of creating a new one.

---

## Returns

### OptunaResult

Returned object containing:

- **best_params** : Dict[str, Any]  
  Best hyperparameter set found by Optuna.

- **best_value** : float  
  Best objective value achieved.

- **study** : optuna.study.Study  
  The underlying Optuna study for advanced inspection.

- **best_run** : RunResult  
  The `RunResult` corresponding to the best trial.

---

## Common training keys

Because each trial delegates to `experiments.run(...)`, the training output keys mirror the underlying algorithm.

### Planner algorithms (`vi`, `pi`)

- `V`
- `V_track`
- `pi`

### Tabular RL algorithms (`q_learning`, `sarsa`)

- `Q`
- `V`
- `pi`
- `Q_track`
- `pi_track`
- `rewards`

---

## Notes on wrappers and transition matrices

This search entrypoint uses the same environment and wrapper logic as `experiments.run(...)`.

- If `env.P` (or `env.unwrapped.P`) exists, it is used directly.
- Otherwise, a wrapper may be applied explicitly or via the internal registry.
- If a valid transition dictionary cannot be obtained, the trial raises an error.

---

## Optional dependency behavior

Optuna is optional.

- Importing `bettermdptools.experiments` does not require Optuna.
- Calling `optimize(...)` without Optuna installed raises `MissingOptunaDependency`.

---

## Stability notes

This entrypoint is intended to be helpful and lightweight.

The most stable contract is:

- `optimize(...)` returns an `OptunaResult`
- `OptunaResult.best_run` is a valid `RunResult`
- `study` is a standard Optuna study for advanced usage

---

## Examples

[`../../examples/optuna_search_examples.ipynb](../../examples/optuna_search_examples.ipynb) 
