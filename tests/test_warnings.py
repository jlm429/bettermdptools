import warnings

import gymnasium as gym
from gymnasium import Wrapper

from bettermdptools.utils.grid_search import GridSearch
from compare_algorithms import run_value_iteration


class WarnOnPWrapper(Wrapper):
    """Wrapper that emits a warning when env.P is accessed."""

    def __getattr__(self, item):
        if item == "P":
            warnings.warn("env.P deprecated in wrapper", UserWarning)
        return super().__getattr__(item)


def test_vi_grid_search_avoids_env_p_warning():
    """Grid search should avoid accessing P"""
    base_env = gym.make("FrozenLake8x8-v1", render_mode=None)
    wrapped_env = WarnOnPWrapper(base_env)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*env\\.P deprecated in wrapper.*", category=UserWarning)
        # unrelated warning for short runs
        warnings.filterwarnings("ignore", message="Max iterations reached before convergence.*")

        GridSearch.vi_grid_search(wrapped_env, gamma=[1.0], n_iters=[3], theta=[1e-3], verbose=False)


def test_compare_algorithms_avoids_env_p_warning():
    """Helper should avoid env.P on wrappers."""
    base_env = gym.make("FrozenLake8x8-v1", render_mode=None)
    wrapped_env = WarnOnPWrapper(base_env)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*env\\.P deprecated in wrapper.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="Max iterations reached before convergence.*")

        run_value_iteration(wrapped_env, n_iters=3, vectorized=False)
