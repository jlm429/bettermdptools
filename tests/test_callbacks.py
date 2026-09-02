from dataclasses import fields

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.algorithms import rl as rl_module
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.callbacks import (
    Callbacks,
    EnvStepContext,
    EpisodeBeginContext,
    EpisodeEndContext,
    ExampleCallbacks,
    MyCallbacks,
)


class CallbackLifecycleEnv(gym.Env):
    observation_space = gym.spaces.Discrete(3)
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.episode = -1
        self.step_index = 0
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_index = 0
        self.reset_count += 1
        return 0, {"reset": self.episode}

    def step(self, action):
        assert self.action_space.contains(action)
        info = {"episode": self.episode, "step": self.step_index}
        if self.episode == 0 and self.step_index == 0:
            transition = (1, 1.0, False, False, info)
        elif self.episode == 0:
            transition = (2, 2.0, True, False, info)
        else:
            transition = (1, 3.0, False, True, info)
        self.step_index += 1
        return transition


class RecordingCallbacks(Callbacks):
    def __init__(self):
        self.events = []
        self.begin_contexts = []
        self.step_contexts = []
        self.end_contexts = []
        self.q_snapshots = []
        self.begin_reset_counts = []

    def on_episode_begin(self, context):
        self.events.append(("begin", context.episode))
        self.begin_contexts.append(context)
        self.begin_reset_counts.append(context.caller.env.reset_count)

    def on_env_step(self, context):
        self.events.append(("step", context.episode, context.step))
        self.step_contexts.append(context)
        self.q_snapshots.append(context.q_values.copy())

    def on_episode_end(self, context):
        self.events.append(("end", context.episode))
        self.end_contexts.append(context)


def train_with_callbacks(
    algorithm, callbacks=None, n_episodes=2, assign_after_init=False
):
    if assign_after_init:
        agent = RL(CallbackLifecycleEnv())
        agent.callbacks = callbacks
    else:
        agent = RL(CallbackLifecycleEnv(), callbacks=callbacks)
    agent.select_action = lambda state, Q, epsilon: 0
    result = getattr(agent, algorithm)(
        gamma=0.75,
        init_alpha=1.0,
        min_alpha=0.5,
        alpha_decay_ratio=1.0,
        init_epsilon=0.4,
        min_epsilon=0.2,
        epsilon_decay_ratio=1.0,
        n_episodes=n_episodes,
    )
    return agent, result


def test_callback_contexts_are_hook_specific_and_mycallbacks_is_subclassable():
    assert [field.name for field in fields(EpisodeBeginContext)] == [
        "caller",
        "episode",
        "alpha",
        "epsilon",
        "gamma",
    ]
    assert [field.name for field in fields(EnvStepContext)] == [
        "caller",
        "episode",
        "step",
        "state",
        "action",
        "next_state",
        "reward",
        "terminated",
        "truncated",
        "info",
        "q_values",
    ]
    assert [field.name for field in fields(EpisodeEndContext)] == [
        "caller",
        "episode",
        "total_reward",
        "step_count",
        "terminated",
        "truncated",
    ]

    class CustomCallbacks(MyCallbacks):
        pass

    assert isinstance(CustomCallbacks(), Callbacks)


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_default_callbacks_skip_context_allocation(monkeypatch, algorithm):
    def fail_context_allocation(**kwargs):
        raise AssertionError("no-op callbacks allocated a context")

    monkeypatch.setattr(rl_module, "EpisodeBeginContext", fail_context_allocation)
    monkeypatch.setattr(rl_module, "EnvStepContext", fail_context_allocation)
    monkeypatch.setattr(rl_module, "EpisodeEndContext", fail_context_allocation)

    agent, _ = train_with_callbacks(algorithm, n_episodes=1)

    assert agent.callbacks.__class__ is MyCallbacks


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
@pytest.mark.parametrize("assign_after_init", [False, True])
def test_explicit_exact_mycallbacks_remain_active(algorithm, assign_after_init):
    callbacks = MyCallbacks()
    events = []
    callbacks.on_episode_begin = lambda context: events.append(
        ("begin", context.episode)
    )
    callbacks.on_env_step = lambda context: events.append(
        ("step", context.episode, context.step)
    )
    callbacks.on_episode_end = lambda context: events.append(("end", context.episode))

    train_with_callbacks(
        algorithm,
        callbacks,
        n_episodes=1,
        assign_after_init=assign_after_init,
    )

    assert events == [
        ("begin", 0),
        ("step", 0, 0),
        ("step", 0, 1),
        ("end", 0),
    ]


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_td_callbacks_report_order_payloads_and_pre_update_q(algorithm):
    callbacks = RecordingCallbacks()
    agent, result = train_with_callbacks(algorithm, callbacks)
    Q = result[0]

    assert callbacks.events == [
        ("begin", 0),
        ("step", 0, 0),
        ("step", 0, 1),
        ("end", 0),
        ("begin", 1),
        ("step", 1, 0),
        ("end", 1),
    ]
    assert callbacks.begin_reset_counts == [0, 1]
    assert all(context.caller is agent for context in callbacks.begin_contexts)
    assert [context.episode for context in callbacks.begin_contexts] == [0, 1]
    np.testing.assert_allclose(
        [context.alpha for context in callbacks.begin_contexts], [1.0, 0.5]
    )
    np.testing.assert_allclose(
        [context.epsilon for context in callbacks.begin_contexts], [0.4, 0.2]
    )
    assert [context.gamma for context in callbacks.begin_contexts] == [0.75, 0.75]

    assert all(context.caller is agent for context in callbacks.step_contexts)
    assert [
        (
            context.episode,
            context.step,
            context.state,
            context.action,
            context.next_state,
            context.reward,
            context.terminated,
            context.truncated,
            context.info,
        )
        for context in callbacks.step_contexts
    ] == [
        (0, 0, 0, 0, 1, 1.0, False, False, {"episode": 0, "step": 0}),
        (0, 1, 1, 0, 2, 2.0, True, False, {"episode": 0, "step": 1}),
        (1, 0, 0, 0, 1, 3.0, False, True, {"episode": 1, "step": 0}),
    ]
    np.testing.assert_array_equal(callbacks.q_snapshots[0], np.zeros((3, 1)))
    np.testing.assert_array_equal(callbacks.q_snapshots[1].ravel(), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(callbacks.q_snapshots[2].ravel(), [1.0, 2.0, 0.0])
    assert Q[0, 0] == 2.75

    assert all(context.caller is agent for context in callbacks.end_contexts)
    assert [
        (
            context.episode,
            context.total_reward,
            context.step_count,
            context.terminated,
            context.truncated,
        )
        for context in callbacks.end_contexts
    ] == [
        (0, 3.0, 2, True, False),
        (1, 3.0, 1, False, True),
    ]


class RaisingCallbacks(Callbacks):
    def __init__(self, failing_hook):
        self.failing_hook = failing_hook
        self.events = []

    def record_or_raise(self, hook):
        self.events.append(hook)
        if hook == self.failing_hook:
            raise RuntimeError(f"failure from {hook}")

    def on_episode_begin(self, context):
        self.record_or_raise("begin")

    def on_env_step(self, context):
        self.record_or_raise("step")

    def on_episode_end(self, context):
        self.record_or_raise("end")


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
@pytest.mark.parametrize(
    ("failing_hook", "expected_events"),
    [
        ("begin", ["begin"]),
        ("step", ["begin", "step"]),
        ("end", ["begin", "step", "step", "end"]),
    ],
)
def test_callback_exceptions_propagate_without_a_later_episode_end(
    algorithm, failing_hook, expected_events
):
    callbacks = RaisingCallbacks(failing_hook)

    with pytest.raises(RuntimeError, match=f"failure from {failing_hook}"):
        train_with_callbacks(algorithm, callbacks, n_episodes=1)

    assert callbacks.events == expected_events


def test_example_callbacks_print_supplied_training_values(capsys):
    callbacks = ExampleCallbacks(log_every=1)

    train_with_callbacks("q_learning", callbacks, n_episodes=1)

    assert capsys.readouterr().out == (
        "[episode 0] epsilon=0.4000 alpha=1.0000 gamma=0.7500\n"
    )
