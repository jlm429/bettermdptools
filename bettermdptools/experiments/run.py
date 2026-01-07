"""
bettermdptools.experiments.run

High-level experiment entry points.

Primary user-facing function: `run(...)`.

An optional ExperimentBuilder is also provided to support fluent,
step-by-step configuration.

"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .env_factory import EnvFactory
from .algorithms import normalize_algo_name, run_algorithm
from .types import EnvBundle, RunResult


def _maybe_set_seed(seed: Optional[int]) -> Optional[int]:
    try:
        # Preferred import location for seeding utilities
        from bettermdptools.utils.seed import set_seed

        return set_seed(seed)
    except Exception:
        # Seeding is best-effort and should not prevent execution
        return seed


def _eval_policy(
    bundle: EnvBundle, pi: Any, eval_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate a policy using TestEnv.test_env, when available."""
    # Attempt known import locations for TestEnv
    TestEnv = None
    for modpath in (
        "bettermdptools.utils.test_env",
        "bettermdptools.test_env",
        "bettermdptools.utils.TestEnv",
    ):
        try:
            module = __import__(modpath, fromlist=["TestEnv"])
            TestEnv = getattr(module, "TestEnv")
            break
        except Exception:
            continue

    if TestEnv is None:
        raise ImportError(
            "TestEnv could not be imported. Expected it at "
            "bettermdptools.utils.test_env.TestEnv or a compatible location."
        )

    scores = TestEnv.test_env(
        bundle.env,
        pi=pi,
        convert_state_obs=bundle.convert_state_obs,
        **eval_kwargs,
    )
    return {"scores": scores}


def run(
    algo: str,
    env_id: str,
    *,
    seed: Optional[int] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    wrapper: Optional[Any] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    algo_kwargs: Optional[Dict[str, Any]] = None,
    eval_kwargs: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Run a single experiment consisting of an environment and algorithm.

    An optional evaluation step can be performed after training.

    Parameters
    ----------
    algo:
        Algorithm name. Supported values include "vi", "pi", "q_learning" ("q"),
        and "sarsa".
    env_id:
        Gymnasium environment id string.
    seed:
        Global seed to apply on a best-effort basis. Note that gymnasium
        environments may not apply seeds consistently.
    env_kwargs:
        Keyword arguments forwarded to gym.make(env_id, **env_kwargs).
    wrapper:
        Optional environment wrapper to apply when the environment does not
        expose a P matrix. This may be a callable or a string import path such as
        "bettermdptools.envs.cartpole_wrapper:CartpoleWrapper".
    wrapper_kwargs:
        Keyword arguments forwarded to the wrapper constructor.
    algo_kwargs:
        Keyword arguments forwarded to the selected algorithm implementation.
    eval_kwargs:
        If provided, the learned policy is evaluated using TestEnv.test_env.

    Returns
    -------
    RunResult
        A dataclass containing training results, optional evaluation output,
        and metadata.
    """
    env_kwargs = env_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    algo_kwargs = algo_kwargs or {}

    used_seed = _maybe_set_seed(seed)

    # EnvFactory maintains a small internal registry, so an instance is used
    factory = EnvFactory()
    bundle = factory.make(
        env_id,
        gym_kwargs=env_kwargs,
        wrapper=wrapper,
        wrapper_kwargs=wrapper_kwargs,
    )

    train_out = run_algorithm(algo, bundle, **algo_kwargs)

    eval_out = None
    if eval_kwargs:
        if "pi" not in train_out:
            raise ValueError("Evaluation requires a policy `pi` in training output.")
        eval_out = _eval_policy(bundle, train_out["pi"], eval_kwargs)

    return RunResult(
        algo=normalize_algo_name(algo),
        env_id=env_id,
        seed=used_seed,
        train=train_out,
        eval=eval_out,
        meta={"env": bundle.meta},
    )


class ExperimentBuilder:
    """builder for configuring and running an experiment.

    This class provides a convenience interface on top of `run(...)`.
    """

    def __init__(self):
        self._algo: Optional[str] = None
        self._env_id: Optional[str] = None
        self._seed: Optional[int] = None
        self._env_kwargs: Dict[str, Any] = {}
        self._wrapper: Optional[Any] = None
        self._wrapper_kwargs: Dict[str, Any] = {}
        self._algo_kwargs: Dict[str, Any] = {}
        self._eval_kwargs: Optional[Dict[str, Any]] = None

    def seed(self, seed: Optional[int]) -> "ExperimentBuilder":
        self._seed = seed
        return self

    def env(self, env_id: str, **env_kwargs: Any) -> "ExperimentBuilder":
        self._env_id = env_id
        self._env_kwargs = dict(env_kwargs)
        return self

    def wrapper(self, wrapper: Any, **wrapper_kwargs: Any) -> "ExperimentBuilder":
        self._wrapper = wrapper
        self._wrapper_kwargs = dict(wrapper_kwargs)
        return self

    def algorithm(self, algo: str, **algo_kwargs: Any) -> "ExperimentBuilder":
        self._algo = algo
        self._algo_kwargs = dict(algo_kwargs)
        return self

    def evaluate(self, **eval_kwargs: Any) -> "ExperimentBuilder":
        self._eval_kwargs = dict(eval_kwargs)
        return self

    def run(self) -> RunResult:
        if self._algo is None:
            raise ValueError("ExperimentBuilder requires algorithm(...) to be set.")
        if self._env_id is None:
            raise ValueError("ExperimentBuilder requires env(...) to be set.")

        return run(
            algo=self._algo,
            env_id=self._env_id,
            seed=self._seed,
            env_kwargs=self._env_kwargs,
            wrapper=self._wrapper,
            wrapper_kwargs=self._wrapper_kwargs,
            algo_kwargs=self._algo_kwargs,
            eval_kwargs=self._eval_kwargs,
        )
