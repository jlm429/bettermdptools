# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class Callbacks:
    """
    Callback interface for RL algorithms.

    Notes
    - All hooks accept `caller` so callbacks can read algorithm state (Q table, epsilon, etc.).
    - Hooks accept `**kwargs` for forward compatibility.
    """

    def on_episode_begin(self, caller: Any, *, episode: Optional[int] = None, **kwargs: Any) -> None:
        pass

    def on_episode_end(self, caller: Any, *, episode: Optional[int] = None, **kwargs: Any) -> None:
        pass

    def on_episode(self, caller: Any, episode: int, **kwargs: Any) -> None:
        pass

    def on_env_step(self, caller: Any, **kwargs: Any) -> None:
        pass

# Backward-compatible name (keeps existing imports working)
class MyCallbacks(Callbacks):
    """
    Backward-compatible callback base.

    Override any hook method in a subclass to execute custom logic during training.
    """
    pass

@dataclass
class ExampleCallbacks(Callbacks):
    """
    Example callbacks showing how to override hooks.

    Parameters
    - log_every: print a message every N episodes (0 disables).
    """

    log_every: int = 100

    def on_episode(self, caller: Any, episode: int, **kwargs: Any) -> None:
        if self.log_every and episode % self.log_every == 0:
            # Safe introspection - only print if attribute exists
            eps = getattr(caller, "epsilon", None)
            alpha = getattr(caller, "alpha", None)
            gamma = getattr(caller, "gamma", None)
            msg = f"[episode {episode}]"
            if eps is not None:
                msg += f" epsilon={eps:.4f}"
            if alpha is not None:
                msg += f" alpha={alpha:.4f}"
            if gamma is not None:
                msg += f" gamma={gamma:.4f}"
            print(msg)

    def on_episode_begin(self, caller: Any, *, episode: Optional[int] = None, **kwargs: Any) -> None:
        # Example: reset per-episode counters stored on the callback instance
        # self.steps_this_episode = 0
        pass

    def on_episode_end(self, caller: Any, *, episode: Optional[int] = None, **kwargs: Any) -> None:
        # Example: read metrics passed by the algorithm if available
        # total_reward = kwargs.get("episode_reward")
        pass

    def on_env_step(self, caller: Any, **kwargs: Any) -> None:
        # Example: track steps, or inspect step-level info if passed in kwargs
        # r = kwargs.get("reward")
        # done = kwargs.get("done")
        pass