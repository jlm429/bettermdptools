# -*- coding: utf-8 -*-
"""
Typed callback contexts for model-free reinforcement-learning algorithms.

Q-learning and SARSA call the hooks in this order for each successful episode:
`on_episode_begin`, one or more `on_env_step` calls, and `on_episode_end`.
The environment-step hook observes a transition before its TD update. Callback
exceptions propagate to the caller, and `on_episode_end` is not a cleanup hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bettermdptools.algorithms.rl import RL


@dataclass(frozen=True, slots=True)
class EpisodeBeginContext:
    """Values available immediately before an episode's environment reset.

    Attributes
    ----------
    caller : RL
        The training object invoking the callback.
    episode : int
        Zero-based episode index.
    alpha : float
        Learning rate selected for this episode.
    epsilon : float
        Exploration rate selected for this episode.
    gamma : float
        Discount factor for this training run.
    """

    caller: RL
    episode: int
    alpha: float
    epsilon: float
    gamma: float


@dataclass(frozen=True, slots=True)
class EnvStepContext:
    """An environment transition observed before its TD update.

    `state` and `next_state` are the tabular indices produced by the training
    call's `convert_state_obs` function. At hook entry, `q_values` has not yet
    incorporated this transition. It is the live table and can reflect later
    updates if a callback retains the reference.

    Attributes
    ----------
    caller : RL
        The training object invoking the callback.
    episode : int
        Zero-based episode index.
    step : int
        Zero-based step index within the episode.
    state : int
        Converted state before the transition.
    action : int
        Action supplied to `env.step`.
    next_state : int
        Converted observation returned by `env.step`.
    reward : float
        Reward returned by `env.step`.
    terminated : bool
        Whether the transition reached a terminal state.
    truncated : bool
        Whether the environment stopped the episode without termination.
    info : Mapping[str, Any]
        Information mapping returned by `env.step`.
    q_values : numpy.ndarray
        Live action-value table before the transition's TD update.
    """

    caller: RL
    episode: int
    step: int
    state: int
    action: int
    next_state: int
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    q_values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EpisodeEndContext:
    """The outcome of a successfully completed training episode.

    Attributes
    ----------
    caller : RL
        The training object invoking the callback.
    episode : int
        Zero-based episode index.
    total_reward : float
        Sum of rewards received during the episode.
    step_count : int
        Number of completed environment steps.
    terminated : bool
        Whether the final transition reached a terminal state.
    truncated : bool
        Whether the final transition stopped the episode without termination.
    """

    caller: RL
    episode: int
    total_reward: float
    step_count: int
    terminated: bool
    truncated: bool


class Callbacks:
    """No-op callback interface for Q-learning and SARSA.

    Override any typed hook in a subclass. Exceptions raised by hooks propagate
    from the training method. An episode-end hook runs only after the algorithm
    records a successfully completed episode, so it is not guaranteed cleanup.
    """

    def on_episode_begin(self, context: EpisodeBeginContext) -> None:
        """Run before the episode's environment reset."""
        pass

    def on_env_step(self, context: EnvStepContext) -> None:
        """Run after an environment step and before its TD update."""
        pass

    def on_episode_end(self, context: EpisodeEndContext) -> None:
        """Run after a successfully completed episode is recorded."""
        pass


class MyCallbacks(Callbacks):
    """Name-compatible base implementing only the typed callback contract."""

    pass


@dataclass
class ExampleCallbacks(Callbacks):
    """Print current training parameters at a configured episode interval.

    Parameters
    ----------
    log_every : int
        Print every N episodes. Zero disables logging.
    """

    log_every: int = 100

    def on_episode_begin(self, context: EpisodeBeginContext) -> None:
        """Print the episode's supplied alpha, epsilon, and gamma values."""
        if self.log_every and context.episode % self.log_every == 0:
            print(
                f"[episode {context.episode}]"
                f" epsilon={context.epsilon:.4f}"
                f" alpha={context.alpha:.4f}"
                f" gamma={context.gamma:.4f}"
            )
