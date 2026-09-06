from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.utils import test_env as test_env_module
from bettermdptools.utils.test_env import TestEnv


@pytest.fixture
def lake():
    env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=1)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize("policy_kwargs", [{}, {"pi": None}])
@pytest.mark.parametrize("render", [False, True])
def test_missing_automatic_policy_fails_before_setup(
    lake, monkeypatch, policy_kwargs, render
):
    reset = Mock(wraps=lake.reset)
    step = Mock(wraps=lake.step)
    backend = Mock()
    copy = Mock(return_value=lake)
    monkeypatch.setattr(lake, "reset", reset)
    monkeypatch.setattr(lake, "step", step)
    monkeypatch.setattr(test_env_module, "_require_rendering_backend", backend)
    monkeypatch.setattr(test_env_module, "_copy_for_rendering", copy)

    with pytest.raises(ValueError, match="pi.*required.*user_input=False"):
        TestEnv.test_env(lake, render=render, n_iters=1, seed=417, **policy_kwargs)

    reset.assert_not_called()
    step.assert_not_called()
    backend.assert_not_called()
    copy.assert_not_called()


class IndexablePolicy:
    def __getitem__(self, state):
        assert state == 0
        return 1


@pytest.mark.parametrize("policy", [{0: 1}, np.ones(16, dtype=int), IndexablePolicy()])
def test_automatic_indexable_policies(lake, monkeypatch, policy):
    step = Mock(wraps=lake.step)
    monkeypatch.setattr(lake, "step", step)
    scores = TestEnv.test_env(lake, n_iters=1, pi=policy, seed=417)
    np.testing.assert_array_equal(scores, [0.0])
    step.assert_called_once_with(1)


@pytest.mark.parametrize(
    "policy", [None, {0: 1}, np.ones(16, dtype=int), IndexablePolicy()]
)
def test_interactive_policy_only_suggests(lake, monkeypatch, capsys, policy):
    monkeypatch.setattr("builtins.input", lambda prompt: "0")
    step = Mock(wraps=lake.step)
    monkeypatch.setattr(lake, "step", step)
    scores = TestEnv.test_env(lake, n_iters=1, pi=policy, user_input=True, seed=417)
    np.testing.assert_array_equal(scores, [0.0])
    step.assert_called_once_with(0)
    output = capsys.readouterr().out
    if policy is None:
        assert "policy output" not in output
    else:
        assert "policy output is 1" in output


@pytest.mark.parametrize("policy_kwargs", [{}, {"pi": None}])
def test_zero_episodes_without_policy(lake, monkeypatch, policy_kwargs):
    reset = Mock(wraps=lake.reset)
    monkeypatch.setattr(lake, "reset", reset)
    scores = TestEnv.test_env(lake, n_iters=0, seed=417, **policy_kwargs)
    assert scores.shape == (0,)
    assert scores.dtype == np.dtype("float64")
    reset.assert_not_called()


@pytest.mark.parametrize("count, error", [(-1, ValueError), (1.5, TypeError)])
def test_invalid_episode_count_keeps_existing_failure(lake, count, error):
    with pytest.raises(error) as exc:
        TestEnv.test_env(lake, n_iters=count)
    assert "pi" not in str(exc.value)
