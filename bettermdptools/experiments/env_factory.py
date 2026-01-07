"""bettermdptools.experiments.env_factory

Environment creation and adaptation for ready-to-run experiments.

This layer sits on top of existing environment wrappers. It focuses on producing
an EnvBundle that includes a Gym-style transition dictionary `P`, along with the
metadata needed for planning and tabular reinforcement learning.

Philosophy
- bettermdptools targets environments that expose a transition dictionary `P`.
- If an environment already exposes `P`, it is used directly.
- If not, an optional wrapper may be applied to provide `P` and tabular spaces.
- If `P` cannot be obtained, an error is raised to keep behavior explicit.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym

from .types import EnvBundle


WrapperSpec = Union[None, Callable[..., Any], str]


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
                    # Wrapper mappings used for environments that do not expose `P`
                    "CartPole": "bettermdptools.envs.cartpole_wrapper:CartpoleWrapper",
                    "Blackjack": "bettermdptools.envs.blackjack_wrapper:BlackjackWrapper",
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
            Keyword arguments forwarded to `gym.make`.
        wrapper:
            Optional wrapper to apply when the environment does not expose `P`.
            Accepted forms:
            - callable: used directly
            - string: resolved via import ("pkg.mod:Class" or "pkg.mod.Class")
            If None, a small built-in registry is consulted.
        wrapper_kwargs:
            Keyword arguments forwarded to the wrapper constructor.

        Returns
        -------
        EnvBundle
        """
        gym_kwargs = dict(gym_kwargs or {})
        wrapper_kwargs = dict(wrapper_kwargs or {})

        env = gym.make(env_id, **gym_kwargs)

        # Prefer env.P; fall back to env.unwrapped.P
        P = getattr(env, "P", None) or getattr(
            getattr(env, "unwrapped", None), "P", None
        )

        if P is not None:
            nS = getattr(getattr(env, "observation_space", None), "n", None)
            nA = getattr(getattr(env, "action_space", None), "n", None)
            if nS is None or nA is None:
                raise ValueError(
                    f"Environment {env_id!r} exposes P but does not have Discrete observation/action spaces. "
                    "Tabular algorithms require Discrete spaces or a wrapper that discretizes them."
                )
            return EnvBundle(
                env=env,
                P=P,
                convert_state_obs=lambda s: s,
                nS=int(nS),
                nA=int(nA),
                meta={"source": "gym", "wrapped": False},
            )

        # If no P is available, optionally apply a wrapper (explicit or registry-based)
        if wrapper is None:
            for key, spec in self._registry.items():
                if key in env_id:
                    wrapper = spec
                    break

        wrapper_callable = _resolve_wrapper(wrapper)
        if wrapper_callable is None:
            raise ValueError(
                f"Environment {env_id!r} does not expose a P matrix, and no wrapper was provided. "
                "This library focuses on environments that provide P for planning/tabular RL. "
                "Provide `wrapper=` (callable or import path) or use a supported env."
            )

        wrapped_env = wrapper_callable(env, **wrapper_kwargs)
        P = getattr(wrapped_env, "P", None) or getattr(
            getattr(wrapped_env, "unwrapped", None), "P", None
        )
        if P is None:
            raise ValueError(
                f"Wrapper {wrapper_callable.__name__} did not expose a P matrix via `.P`. "
                "Supported wrappers should provide a `.P` property."
            )

        convert = getattr(wrapped_env, "transform_obs", None)

        # Some wrappers may transform observations internally.
        # If observations are already integers, conversion should be an identity function.
        if convert is not None:
            obs_space = getattr(wrapped_env, "observation_space", None)
            if getattr(obs_space, "n", None) is not None:
                convert = lambda s: s

        if convert is None:
            convert = lambda s: s

        nS = getattr(getattr(wrapped_env, "observation_space", None), "n", None)
        nA = getattr(getattr(wrapped_env, "action_space", None), "n", None)
        if nS is None or nA is None:
            raise ValueError(
                f"Wrapped environment for {env_id!r} does not expose Discrete observation/action spaces. "
                "Wrappers should set observation_space to Discrete(nS) and preserve action_space."
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
