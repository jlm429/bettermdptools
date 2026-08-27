"""bettermdptools.experiments.env_factory

Environment creation and adaptation for ready-to-run experiments.

This layer sits on top of existing environment wrappers. It focuses on producing
an EnvBundle that includes a Gymnasium-style transition dictionary `P`, along with
metadata needed for planning and tabular reinforcement learning.

Philosophy
- bettermdptools targets environments with discrete spaces and a transition
  dictionary `P`.
- Built-in models are read from `env.unwrapped.P`.
- Explicit wrapper models are read from the wrapper's `env.P` property.
- A wrapper may be applied when the native model or spaces are not usable.
- If a usable tabular model cannot be obtained, an error is raised.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym

from .types import EnvBundle

WrapperSpec = Union[None, Callable[..., Any], str]


def _identity(value: Any) -> Any:
    return value


def _get_transition_model(env: gym.Env) -> Any:
    """Return a tabular model without relying on wrapper attribute forwarding."""
    model = getattr(env, "P", None)
    if model is not None:
        return model
    return getattr(env.unwrapped, "P", None)


def _discrete_space_sizes(env: gym.Env) -> Tuple[Optional[int], Optional[int]]:
    """Return tabular space sizes when both environment spaces are discrete."""
    nS = getattr(getattr(env, "observation_space", None), "n", None)
    nA = getattr(getattr(env, "action_space", None), "n", None)
    if nS is None or nA is None:
        return None, None
    return int(nS), int(nA)


def _get_attr_chain(obj: Any, names: Tuple[str, ...]) -> Any:
    for n in names:
        if obj is None:
            return None
        obj = getattr(obj, n, None)
    return obj


def _resolve_wrapper(wrapper: WrapperSpec) -> Optional[Callable[..., Any]]:
    """Resolve a wrapper from a callable or an import path string.

    If wrapper is a string, accepted formats are:
    - "package.module:ClassName" (preferred)
    - "package.module.ClassName"
    """
    if wrapper is None:
        return None
    if callable(wrapper):
        return wrapper
    if not isinstance(wrapper, str):
        raise TypeError("wrapper must be None, a callable, or an import path string")

    if ":" in wrapper:
        mod, cls = wrapper.split(":", 1)
    else:
        mod, cls = wrapper.rsplit(".", 1)

    module = importlib.import_module(mod)
    resolved = getattr(module, cls)
    if not callable(resolved):
        raise TypeError(f"Resolved wrapper {wrapper!r} is not callable")
    return resolved


@dataclass(frozen=True)
class EnvFactory:
    """Factory that creates an EnvBundle.

    Additional wrappers can be supported by passing `wrapper=` without changing
    this module.
    """

    # Built-in minimal registry.
    # Matching strategy: if env_id contains the key substring, apply the wrapper.
    _registry: Dict[str, str] = None

    def __post_init__(self):
        if self._registry is None:
            object.__setattr__(
                self,
                "_registry",
                {
                    # Wrapper mappings for environments without a usable
                    # native model.
                    "CartPole": (
                        "bettermdptools.envs.cartpole_wrapper:CartpoleWrapper"
                    ),
                    "Blackjack": (
                        "bettermdptools.envs.blackjack_wrapper:BlackjackWrapper"
                    ),
                    "Acrobot": "bettermdptools.envs.acrobot_wrapper:AcrobotWrapper",
                    "Pendulum": "bettermdptools.envs.pendulum_wrapper:PendulumWrapper",
                },
            )

    def make(
        self,
        env_id: str,
        *,
        gym_kwargs: Optional[Dict[str, Any]] = None,
        wrapper: WrapperSpec = None,
        wrapper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> EnvBundle:
        """Create an EnvBundle.

        Parameters
        ----------
        env_id:
            Gymnasium environment id (for example, "FrozenLake8x8-v1", "CartPole-v1").
        gym_kwargs:
            Keyword arguments forwarded to `gymnasium.make`.
        wrapper:
            Optional wrapper to apply when the environment does not expose a usable
            tabular model and discrete observation and action spaces.
            Accepted forms:
            - callable: used directly
            - string: resolved via import ("pkg.mod:Class" or "pkg.mod.Class")
            If None, a small built-in registry is consulted.
        wrapper_kwargs:
            Keyword arguments forwarded to the wrapper constructor.

        Returns
        -------
        EnvBundle
            The caller owns the returned environment and must close it.
        """
        gym_kwargs = dict(gym_kwargs or {})
        wrapper_kwargs = dict(wrapper_kwargs or {})

        env = gym.make(env_id, **gym_kwargs)
        owned_env = env

        try:
            P = _get_transition_model(env)

            nS, nA = _discrete_space_sizes(env)
            native_reason = (
                "it does not expose native P"
                if P is None
                else "its observation and action spaces are not both Discrete"
            )

            if P is not None and nS is not None and nA is not None:
                return EnvBundle(
                    env=env,
                    P=P,
                    convert_state_obs=_identity,
                    nS=int(nS),
                    nA=int(nA),
                    meta={"source": "gym", "wrapped": False},
                )

            # If no P is available, optionally apply a wrapper
            if wrapper is None:
                for key, spec in self._registry.items():
                    if key in env_id:
                        wrapper = spec
                        break

            wrapper_callable = _resolve_wrapper(wrapper)
            if wrapper_callable is None:
                raise ValueError(
                    f"Environment {env_id!r} cannot provide a tabular transition "
                    "model: the native-P route is unavailable because "
                    f"{native_reason}; "
                    "the wrapper-P route is unavailable because no explicit or "
                    "registered bettermdptools wrapper was found. Use an environment "
                    "with native P and Discrete observation/action spaces, or provide "
                    "a supported `wrapper=` that exposes P and Discrete spaces."
                )

            wrapped_env = wrapper_callable(env, **wrapper_kwargs)
            owned_env = wrapped_env
            P = _get_transition_model(wrapped_env)
            if P is None:
                raise ValueError(
                    f"Environment {env_id!r} cannot provide a tabular transition "
                    "model: the native-P route is unavailable because "
                    f"{native_reason}; "
                    f"the wrapper-P route is unavailable because wrapper "
                    f"{wrapper_callable.__name__} did not expose P."
                )

            convert = getattr(wrapped_env, "transform_obs", None)

            # Some wrappers may transform observations internally.
            # Integer observations need only an identity conversion.
            if convert is not None:
                obs_space = getattr(wrapped_env, "observation_space", None)
                if getattr(obs_space, "n", None) is not None:
                    convert = _identity

            if convert is None:
                convert = _identity

            nS = getattr(getattr(wrapped_env, "observation_space", None), "n", None)
            nA = getattr(getattr(wrapped_env, "action_space", None), "n", None)
            if nS is None or nA is None:
                raise ValueError(
                    f"Environment {env_id!r} cannot provide a tabular transition "
                    "model: the native-P route is unavailable because "
                    f"{native_reason}; "
                    f"the wrapper-P route is unavailable because wrapper "
                    f"{wrapper_callable.__name__} does not expose Discrete observation "
                    "and action spaces."
                )

            return EnvBundle(
                env=wrapped_env,
                P=P,
                convert_state_obs=convert,
                nS=int(nS),
                nA=int(nA),
                meta={
                    "source": "wrapped",
                    "wrapped": True,
                    "wrapper": getattr(wrapper_callable, "__name__", str(wrapper)),
                },
            )
        except Exception:
            owned_env.close()
            raise
