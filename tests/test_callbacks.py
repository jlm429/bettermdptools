from dataclasses import FrozenInstanceError

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.callbacks import (
    Callbacks,
    EpisodeContext,
    MyCallbacks,
    TransitionContext,
)


class TwoStepEnv(gym.Env):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, boundary):
        self.boundary = boundary
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return 0, {"phase": "reset"}

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        if self.steps == 1:
            return 1, 1.0, False, False, {"transition": 1}
        return (
            0,
            2.0,
            self.boundary == "terminated",
            self.boundary == "truncated",
            {"transition": 2},
        )


class ReusedObservationEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.int64)
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.observation = np.zeros(1, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation[0] = 0
        return self.observation, {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.observation[0] = 1
        return self.observation, 1.0, True, False, {}


class ReusedInfoEnv(gym.Env):
    observation_space = gym.spaces.Discrete(3)
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.steps = 0
        self.metrics = np.zeros(1, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.metrics[0] = 0
        return 0, {"metrics": self.metrics}

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        self.metrics[0] = self.steps
        return (
            self.steps,
            0.0,
            self.steps == 2,
            False,
            {"metrics": self.metrics},
        )


class ContextRecorder(Callbacks):
    def __init__(self):
        self.events = []
        self.episodes = []
        self.transitions = []

    def on_episode_begin(self, caller, *, context):
        self.events.append("begin")
        self.episodes.append(context)

    def on_episode(self, caller, *, context):
        self.events.append("episode")
        assert context.episode == 0

    def on_env_step(self, caller, *, context):
        self.events.append("step")
        self.transitions.append(context)
        return False

    def on_episode_end(self, caller, *, context):
        self.events.append("end")
        self.episodes.append(context)


@pytest.mark.parametrize(
    ("algorithm", "boundary", "injection"),
    (
        ("q_learning", "truncated", "constructor"),
        ("sarsa", "terminated", "method"),
    ),
)
def test_callbacks_receive_ordered_typed_contexts(
    algorithm,
    boundary,
    injection,
):
    env = TwoStepEnv(boundary)
    recorder = ContextRecorder()
    learner = RL(env, callbacks=recorder if injection == "constructor" else None)
    kwargs = {
        "n_episodes": 1,
        "init_epsilon": 0.0,
        "min_epsilon": 0.0,
        "seed": 417,
    }
    if injection == "method":
        kwargs["callbacks"] = recorder

    result = getattr(learner, algorithm)(**kwargs)

    assert recorder.events == ["begin", "episode", "step", "step", "end"]
    assert all(isinstance(item, EpisodeContext) for item in recorder.episodes)
    assert all(isinstance(item, TransitionContext) for item in recorder.transitions)
    start, end = recorder.episodes
    first, second = recorder.transitions
    assert start.algorithm == algorithm
    assert start.episode == 0
    assert start.episode_number == 1
    assert start.total_episodes == 1
    assert start.progress == 1.0
    assert start.observation == 0
    assert start.state == 0
    assert start.info == {"phase": "reset"}
    assert start.steps == 0
    assert start.total_reward == 0.0
    assert not start.done
    assert first.step == 1
    assert first.observation == 0
    assert first.action == 0
    assert first.reward == 1.0
    assert first.next_observation == 1
    assert first.next_state == 1
    assert first.info == {"transition": 1}
    assert first.total_reward == 1.0
    assert not first.done
    assert second.step == 2
    assert second.observation == 1
    assert second.next_observation == 0
    assert second.total_reward == 3.0
    assert second.terminated is (boundary == "terminated")
    assert second.truncated is (boundary == "truncated")
    assert second.done
    assert end.observation == 0
    assert end.state == 0
    assert end.info == {"transition": 2}
    assert end.steps == 2
    assert end.total_reward == 3.0
    assert end.terminated is second.terminated
    assert end.truncated is second.truncated
    np.testing.assert_array_equal(result[-1], [3.0])

    with pytest.raises(FrozenInstanceError):
        start.episode = 2
    with pytest.raises(TypeError):
        start.info["phase"] = "changed"


class NamedCallbacks:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.transition_data = []

    def on_episode_begin(self, caller, *, context):
        self.events.append((self.name, "begin"))

    def on_episode(self, caller, *, context):
        self.events.append((self.name, f"episode-{context.episode}"))

    def on_env_step(self, caller, *, context):
        self.events.append((self.name, "step"))
        self.transition_data.append(
            (context.reward, context.done, context.total_reward)
        )

    def on_episode_end(self, caller, *, context):
        self.events.append((self.name, "end"))


def test_named_hooks_and_ordered_callback_iterables_run_in_injection_order():
    events = []
    first = NamedCallbacks("first", events)
    second = NamedCallbacks("second", events)
    learner = RL(TwoStepEnv("terminated"))
    learner.callbacks = [first, second]

    learner.q_learning(
        n_episodes=1,
        init_epsilon=0.0,
        min_epsilon=0.0,
    )

    assert events == [
        ("first", "begin"),
        ("second", "begin"),
        ("first", "episode-0"),
        ("second", "episode-0"),
        ("first", "step"),
        ("second", "step"),
        ("first", "step"),
        ("second", "step"),
        ("first", "end"),
        ("second", "end"),
    ]
    assert first.transition_data == [(1.0, False, 1.0), (2.0, True, 3.0)]
    assert second.transition_data == first.transition_data


def test_constructor_callback_generator_is_reused_across_training_calls():
    algorithms = []

    class AlgorithmRecorder(Callbacks):
        def on_episode_begin(self, caller, *, context):
            algorithms.append(context.algorithm)

    recorder = AlgorithmRecorder()
    learner = RL(
        TwoStepEnv("terminated"),
        callbacks=(callback for callback in (recorder,)),
    )

    learner.q_learning(n_episodes=1, init_epsilon=0.0, min_epsilon=0.0)
    learner.sarsa(n_episodes=1, init_epsilon=0.0, min_epsilon=0.0)

    assert learner.callbacks == (recorder,)
    assert algorithms == ["q_learning", "sarsa"]


@pytest.mark.parametrize("algorithm", ("q_learning", "sarsa"))
def test_callback_contexts_snapshot_reused_observation_buffers(algorithm):
    env = ReusedObservationEnv()
    recorder = ContextRecorder()
    learner = RL(env, callbacks=recorder)

    getattr(learner, algorithm)(
        nS=2,
        nA=1,
        convert_state_obs=lambda observation: int(observation[0]),
        n_episodes=1,
        init_epsilon=0.0,
        min_epsilon=0.0,
    )

    start, end = recorder.episodes
    transition = recorder.transitions[0]
    env.observation[0] = 0

    np.testing.assert_array_equal(start.observation, [0])
    np.testing.assert_array_equal(transition.observation, [0])
    np.testing.assert_array_equal(transition.next_observation, [1])
    np.testing.assert_array_equal(end.observation, [1])
    assert not np.shares_memory(start.observation, env.observation)
    assert not np.shares_memory(transition.next_observation, env.observation)
    assert not np.shares_memory(end.observation, env.observation)


@pytest.mark.parametrize("algorithm", ("q_learning", "sarsa"))
def test_callback_contexts_snapshot_nested_info_values(algorithm):
    env = ReusedInfoEnv()
    recorder = ContextRecorder()
    learner = RL(env, callbacks=recorder)

    getattr(learner, algorithm)(
        n_episodes=1,
        init_epsilon=0.0,
        min_epsilon=0.0,
    )

    start, end = recorder.episodes
    first, second = recorder.transitions
    env.metrics[0] = 99

    np.testing.assert_array_equal(start.info["metrics"], [0])
    np.testing.assert_array_equal(first.info["metrics"], [1])
    np.testing.assert_array_equal(second.info["metrics"], [2])
    np.testing.assert_array_equal(end.info["metrics"], [2])
    for context in (start, first, second, end):
        assert not np.shares_memory(context.info["metrics"], env.metrics)


def test_callback_type_errors_propagate_without_retry_or_masking():
    expected = TypeError("callback implementation failed")

    class BuggyCallback(MyCallbacks):
        def __init__(self):
            self.calls = 0

        def on_env_step(self, caller, *, context):
            self.calls += 1
            raise expected

    callback = BuggyCallback()

    with pytest.raises(TypeError) as caught:
        RL(TwoStepEnv("terminated")).q_learning(
            n_episodes=1,
            init_epsilon=0.0,
            min_epsilon=0.0,
            callbacks=callback,
        )

    assert caught.value is expected
    assert callback.calls == 1
