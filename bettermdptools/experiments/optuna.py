from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .run import run as run_experiment


class MissingOptunaDependency(ImportError):
    """Raised when Optuna is not installed but an Optuna workflow is invoked."""


def _lazy_import_optuna():
    """Import Optuna only when needed."""
    try:
        import optuna  # type: ignore

        return optuna
    except Exception as e:  # pragma: no cover
        raise MissingOptunaDependency(
            "Optuna is not installed. Install with `poetry install --with optuna` "
            "or an equivalent optional dependency install."
        ) from e


@dataclass
class OptunaResult:
    """Convenience wrapper for Optuna study results."""

    best_params: Dict[str, Any]
    best_value: float
    study: Any  # optuna.study.Study
    best_run: Any  # bettermdptools.experiments.types.RunResult (or compatible)


def optimize(
    *,
    algo: str,
    env_id: str,
    metric: Callable[[Any], float],
    suggest: Callable[[Any], Dict[str, Dict[str, Any]]],
    n_trials: int = 50,
    seed: Optional[int] = None,
    base_env_kwargs: Optional[Dict[str, Any]] = None,
    base_wrapper_kwargs: Optional[Dict[str, Any]] = None,
    base_algo_kwargs: Optional[Dict[str, Any]] = None,
    base_eval_kwargs: Optional[Dict[str, Any]] = None,
    direction: str = "maximize",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = False,
) -> OptunaResult:
    """Optimize hyperparameters using Optuna, calling `experiments.run(...)` per trial.

    The `suggest(trial)` callable should return a dict with any of:
        - env_kwargs
        - wrapper_kwargs
        - algo_kwargs
        - eval_kwargs
    """
    optuna = _lazy_import_optuna()

    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'")

    base_env_kwargs = base_env_kwargs or {}
    base_wrapper_kwargs = base_wrapper_kwargs or {}
    base_algo_kwargs = base_algo_kwargs or {}
    base_eval_kwargs = base_eval_kwargs or {}

    best_run_holder: Dict[str, Any] = {"run": None, "value": None}

    def objective(trial):
        proposed = suggest(trial) or {}

        env_kwargs = {**base_env_kwargs, **proposed.get("env_kwargs", {})}
        wrapper_kwargs = {**base_wrapper_kwargs, **proposed.get("wrapper_kwargs", {})}
        algo_kwargs = {**base_algo_kwargs, **proposed.get("algo_kwargs", {})}
        eval_kwargs = {**base_eval_kwargs, **proposed.get("eval_kwargs", {})}

        out = run_experiment(
            algo=algo,
            env_id=env_id,
            seed=seed,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
            algo_kwargs=algo_kwargs,
            eval_kwargs=eval_kwargs if eval_kwargs else None,
        )

        value = float(metric(out))

        if best_run_holder["run"] is None:
            best_run_holder["run"] = out
            best_run_holder["value"] = value
        else:
            best_val = float(best_run_holder["value"])
            is_better = (
                value > best_val if direction == "maximize" else value < best_val
            )
            if is_better:
                best_run_holder["run"] = out
                best_run_holder["value"] = value

        return value

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(objective, n_trials=n_trials)

    return OptunaResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        study=study,
        best_run=best_run_holder["run"],
    )
