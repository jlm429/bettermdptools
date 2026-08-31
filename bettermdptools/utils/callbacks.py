# -*- coding: utf-8 -*-
"""Typed callback contexts and explicit dispatch for model-free training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """A read-only snapshot at the start or end of a training episode.

    episode is zero-based, while episode_number is its one-based convenience
    form. At episode start, steps and total_reward are zero and both boundary
    flags are false. At episode end, observation, state, and info describe the
    final environment transition.
    """

    algorithm: str
    episode: int
    total_episodes: int
    observation: Any
    state: Any
    info: Mapping[str, Any]
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    epsilon: float
    alpha: float
    gamma: float

    @property
    def episode_number(self) -> int:
        """Return the one-based episode number."""
        return self.episode + 1

    @property
    def progress(self) -> float:
        """Return this episode's one-based position as a fraction of the run."""
        return self.episode_number / self.total_episodes

    @property
    def done(self) -> bool:
        """Return whether termination or truncation ended the episode."""
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class TransitionContext:
    """A read-only snapshot immediately after an environment transition.

    step is the one-based transition count within the episode. total_reward
    includes this transition's reward. The callback runs before the tabular TD
    update.
    """

    algorithm: str
    episode: int
    total_episodes: int
    step: int
    observation: Any
    state: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    next_observation: Any
    next_state: Any
    info: Mapping[str, Any]
    total_reward: float
    epsilon: float
    alpha: float
    gamma: float

    @property
    def episode_number(self) -> int:
        """Return the one-based episode number."""
        return self.episode + 1

    @property
    def progress(self) -> float:
        """Return this episode's one-based position as a fraction of the run."""
        return self.episode_number / self.total_episodes

    @property
    def done(self) -> bool:
        """Return whether termination or truncation ended the episode."""
        return self.terminated or self.truncated


class Callbacks:
    """Callback interface for Q-learning and SARSA.

    Episode-start hooks run after reset and state conversion, before the first
    action. on_episode_begin runs before the on_episode hook.
    Transition hooks run after env.step and reward accumulation, before the TD
    update. Episode-end hooks run after episode tracking is recorded.

    A hook return value is ignored. Context fields cannot be reassigned, and
    callback code should also treat nested observations as read-only. Callback
    exceptions propagate unchanged and stop training. When multiple callbacks
    are injected, each hook runs in the order supplied.

    Every hook has one explicit signature: ``hook(caller, *, context)``.
    Dispatch does not inspect signatures or catch callback exceptions.
    """

    def on_episode_begin(
        self,
        caller: Any,
        *,
        context: EpisodeContext,
    ) -> None:
        """Run after reset and before the first episode action."""

    def on_episode_end(
        self,
        caller: Any,
        *,
        context: EpisodeContext,
    ) -> None:
        """Run after the completed episode has been recorded."""

    def on_episode(
        self,
        caller: Any,
        *,
        context: EpisodeContext,
    ) -> None:
        """Run after on_episode_begin at the start of each episode."""

    def on_env_step(
        self,
        caller: Any,
        *,
        context: TransitionContext,
    ) -> None:
        """Run after an environment transition and before the TD update."""


class MyCallbacks(Callbacks):
    """Backward-compatible name for the callback base class."""


@dataclass
class ExampleCallbacks(Callbacks):
    """Print schedule metrics at a configured episode interval."""

    log_every: int = 100

    def on_episode(
        self,
        caller: Any,
        *,
        context: EpisodeContext,
    ) -> None:
        """Print selected schedule values at the configured interval."""
        episode = context.episode
        if not self.log_every or episode % self.log_every != 0:
            return

        message = f"[episode {episode}]"
        message += (
            f" epsilon={context.epsilon:.4f}"
            f" alpha={context.alpha:.4f}"
            f" gamma={context.gamma:.4f}"
        )
        print(message)


CallbackHook: TypeAlias = Literal[
    "on_episode_begin",
    "on_episode",
    "on_env_step",
    "on_episode_end",
]
CallbackSpec: TypeAlias = Callbacks | Iterable[Callbacks]


def _snapshot_info(info: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Copy an environment info mapping into a read-only top-level view."""
    return MappingProxyType(dict(info or {}))


def _snapshot_observation(observation: Any) -> Any:
    """Copy mutable observations while preserving immutable values."""
    if observation is None or isinstance(
        observation, (bool, int, float, complex, str, bytes)
    ):
        return observation
    return deepcopy(observation)


def _normalize_callbacks(callbacks: Any) -> tuple[Any, ...]:
    """Normalize one callback or an ordered iterable without consuming it twice."""
    if callbacks is None:
        return ()
    if any(
        callable(getattr(callbacks, hook, None))
        for hook in (
            "on_episode_begin",
            "on_episode",
            "on_env_step",
            "on_episode_end",
        )
    ):
        return (callbacks,)

    try:
        normalized = tuple(callbacks)
    except TypeError as error:
        raise TypeError(
            "callbacks must be a callback object or an iterable of callback objects"
        ) from error

    for callback in normalized:
        if not any(
            callable(getattr(callback, hook, None))
            for hook in (
                "on_episode_begin",
                "on_episode",
                "on_env_step",
                "on_episode_end",
            )
        ):
            raise TypeError("each callback must define at least one callback hook")
    return normalized


def _dispatch_callbacks(
    callbacks: tuple[Any, ...],
    hook: CallbackHook,
    caller: Any,
    context: EpisodeContext | TransitionContext,
) -> None:
    """Invoke one hook on each callback in stable injection order."""
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if callable(method):
            method(caller, context=context)


__all__ = [
    "CallbackSpec",
    "Callbacks",
    "EpisodeContext",
    "ExampleCallbacks",
    "MyCallbacks",
    "TransitionContext",
]
