import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.envs.binning import generate_bin_edges
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.pendulum_discretized import (
    DiscretizedPendulum,
    angle_normalize,
    index_to_continous_state,
    wrap,
)
from bettermdptools.envs.pendulum_wrapper import PendulumWrapper
from bettermdptools.experiments import run
from bettermdptools.experiments.env_factory import EnvFactory
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.test_env import TestEnv


class RenderTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)
    instances = []

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.render_calls = 0
        self.closed = False
        type(self).instances.append(self)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        assert self.action_space.contains(action)
        if self.render_mode == "human":
            self.render()
        return 0, 1.0, True, False, {}

    def render(self):
        self.render_calls += 1

    def close(self):
        self.closed = True


class OffsetObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(2)

    def observation(self, observation):
        return int(observation) + 1


class LifecycleEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)
    P = {0: {0: [(1.0, 0, 0.0, True)]}}
    instances = []

    def __init__(self):
        self.closed = False
        type(self).instances.append(self)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        assert self.action_space.contains(action)
        return 0, 0.0, True, False, {}

    def close(self):
        self.closed = True


class ModelLessLifecycleEnv(LifecycleEnv):
    P = None
    instances = []


def register_test_env(env_id, entry_point):
    if env_id not in gym.registry:
        gym.register(env_id, entry_point=entry_point)


def test_policy_evaluation_bounds_an_undiscounted_recurrent_policy():
    recurrent_model = {0: {0: [(1.0, 0, -1.0, False)]}}
    terminal_model = {0: {0: [(1.0, 0, -1.0, True)]}}

    with pytest.warns(UserWarning, match="policy evaluation converged"):
        recurrent_values = Planner(recurrent_model).policy_evaluation(
            {0: 0}, np.zeros(1), n_iters=3
        )
    with warnings.catch_warnings(record=True) as caught:
        terminal_values = Planner(terminal_model).policy_evaluation(
            {0: 0}, np.zeros(1), n_iters=3
        )

    np.testing.assert_array_equal(recurrent_values, [-3.0])
    np.testing.assert_array_equal(terminal_values, [-1.0])
    assert caught == []

    np.random.seed(0)
    with pytest.warns(UserWarning, match="Max iterations reached"):
        policy_values = Planner(recurrent_model).policy_iteration(
            n_iters=2, eval_n_iters=3
        )[0]
    np.testing.assert_array_equal(policy_values, [-3.0])


def test_policy_iteration_waits_for_stable_policy_evaluation(monkeypatch):
    model = {
        0: {
            0: [(1.0, 0, -1.0, False)],
            1: [(1.0, 0, -1002.0, True)],
        }
    }
    monkeypatch.setattr(
        np.random, "choice", lambda actions, size: np.zeros(size, dtype=int)
    )

    with warnings.catch_warnings(record=True) as caught:
        values, _, policy = Planner(model).policy_iteration(n_iters=4)

    np.testing.assert_array_equal(values, [-1002.0])
    assert policy == {0: 1}
    assert caught == []


def test_taxi_policy_iteration_matches_value_iteration_without_hanging():
    env = gym.make("Taxi-v4")
    try:
        planner = Planner(env.unwrapped.P)
        expected_values, _, expected_policy = planner.value_iteration()

        np.random.seed(0)
        values, _, policy = planner.policy_iteration(eval_n_iters=20)

        np.testing.assert_allclose(values, expected_values)
        assert policy == expected_policy
    finally:
        env.close()


@pytest.mark.parametrize(
    "method_name",
    ["value_iteration", "value_iteration_vectorized", "policy_iteration"],
)
def test_undiscounted_planning_ties_prefer_a_policy_that_reaches_terminal(
    method_name,
):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        np.random.seed(0)
        values, _, policy = getattr(Planner(env.unwrapped.P), method_name)()
        scores = TestEnv.test_env(env, n_iters=2, pi=policy, seed=417)

        assert values[0] == 1.0
        np.testing.assert_array_equal(scores, [1.0, 1.0])
    finally:
        env.close()


def test_undiscounted_terminal_progress_preserves_strict_primary_optimality():
    model = {
        0: {
            0: [(1.0, 0, np.float32(1.0 - 1e-7), True)],
            1: [(1.0, 0, np.float32(1.0), True)],
        }
    }

    policy = Planner(model).policy_improvement(
        np.zeros(1, dtype=np.float32), gamma=1.0, dtype=np.float32
    )

    assert policy == {0: 1}


def test_undiscounted_terminal_progress_has_no_fixed_propagation_horizon():
    terminal_state = 1000
    model = {
        0: {
            0: [(1.0, 0, 0.0, False)],
            1: [(1.0, 1, 0.0, False)],
        }
    }
    for state in range(1, terminal_state):
        transition = [(1.0, state + 1, 0.0, False)]
        model[state] = {0: transition, 1: transition}
    terminal_transition = [(1.0, terminal_state, 0.0, True)]
    model[terminal_state] = {0: terminal_transition, 1: terminal_transition}

    policy = Planner(model).policy_improvement(
        np.zeros(len(model), dtype=np.float32), gamma=1.0, dtype=np.float32
    )

    assert policy[0] == 1


@pytest.mark.parametrize(
    "method_name",
    ["value_iteration", "value_iteration_vectorized", "policy_iteration"],
)
def test_planning_iterations_reject_an_empty_iteration_budget(method_name):
    model = {0: {0: [(1.0, 0, 1.0, True)]}}

    with pytest.raises(ValueError, match="n_iters must be an integer of at least 2"):
        getattr(Planner(model), method_name)(n_iters=1)


@pytest.mark.parametrize(
    ("max_steps", "decay_ratio", "expected"),
    [
        (1, 0.5, [0.5]),
        (2, 0.5, [0.5, 0.01]),
        (2, 0.9, [0.5, 0.01]),
        (10, 0.01, [0.5, *([0.01] * 9)]),
    ],
)
def test_short_decay_schedules_are_finite_and_reach_the_minimum(
    max_steps, decay_ratio, expected
):
    schedule = RL.decay_schedule(0.5, 0.01, decay_ratio, max_steps)

    np.testing.assert_allclose(schedule, expected)
    assert np.isfinite(schedule).all()
    assert np.all(np.diff(schedule) <= 0)


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_short_td_runs_do_not_emit_decay_warnings_or_non_finite_values(algorithm):
    env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=1)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Q = getattr(RL(env), algorithm)(n_episodes=2)[0]

        assert np.isfinite(Q).all()
        assert caught == []
    finally:
        env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decay_ratio": 0},
        {"decay_ratio": 1.1},
        {"max_steps": 0},
        {"init_value": -1, "min_value": 0},
        {"init_value": "invalid"},
    ],
)
def test_decay_schedule_rejects_invalid_parameters(kwargs):
    options = {
        "init_value": 1.0,
        "min_value": 0.1,
        "decay_ratio": 0.5,
        "max_steps": 10,
    }
    options.update(kwargs)

    with pytest.raises(ValueError):
        RL.decay_schedule(**options)


def test_generate_bin_edges_is_finite_symmetric_and_monotonic():
    centered = np.asarray(
        generate_bin_edges(np.float32(2), np.int64(5), np.float32(3), center=True)
    )
    outer_fine = np.asarray(generate_bin_edges(2, 5, 3, center=False))

    assert len(centered) == 6
    assert centered[0] == -2
    assert centered[-1] == 2
    assert np.isfinite(centered).all()
    assert np.all(np.diff(centered) > 0)
    np.testing.assert_allclose(centered, -centered[::-1])
    assert np.diff(centered)[0] > np.diff(centered)[2]
    assert np.diff(outer_fine)[0] < np.diff(outer_fine)[2]


@pytest.mark.parametrize(
    "args",
    [
        (np.inf, 3, 3),
        (1, 3, np.inf),
        (1, 4, 3),
        (1, True, 3),
    ],
)
def test_generate_bin_edges_rejects_invalid_boundaries(args):
    with pytest.raises(ValueError):
        generate_bin_edges(*args)


def test_pendulum_cache_identity_includes_sampling_resolution(tmp_path):
    shared_cache = tmp_path / "shared"
    separate_cache = tmp_path / "separate"

    samples3 = DiscretizedPendulum(
        3, 3, 3, n_workers=1, cache_dir=shared_cache, dim_samples=3
    )
    samples5 = DiscretizedPendulum(
        3, 3, 3, n_workers=1, cache_dir=shared_cache, dim_samples=5
    )
    fresh_samples5 = DiscretizedPendulum(
        3, 3, 3, n_workers=1, cache_dir=separate_cache, dim_samples=5
    )

    assert samples3.P != samples5.P
    assert samples5.P == fresh_samples5.P
    assert len(list(Path(shared_cache).glob("*.pkl.gz"))) == 2


def test_pendulum_parallel_and_serial_construction_are_identical(tmp_path):
    serial = DiscretizedPendulum(
        3, 3, 3, n_workers=1, cache_dir=tmp_path / "serial", dim_samples=3
    )
    parallel = DiscretizedPendulum(
        3, 3, 3, n_workers=2, cache_dir=tmp_path / "parallel", dim_samples=3
    )

    assert parallel.P == serial.P


def test_pendulum_boundaries_wrap_and_indices_reject_aliases(tmp_path):
    model = DiscretizedPendulum(3, 3, 3, n_workers=1, cache_dir=tmp_path, dim_samples=3)

    assert np.isscalar(angle_normalize(1.0))
    np.testing.assert_allclose(
        angle_normalize(np.array([-np.pi, np.pi, 3 * np.pi])),
        [-np.pi, -np.pi, -np.pi],
    )
    assert model.discretize_angle(-np.pi) == 0
    assert model.discretize_angle(np.pi) == 0
    assert model.discretize_angle(3 * np.pi) == 0
    assert model.discretize_angle(np.nextafter(np.pi, 0)) == model.angle_bins - 1
    assert wrap(np.nextafter(np.pi, 0), -np.pi, np.pi) == np.nextafter(np.pi, 0)
    assert wrap(np.pi, -np.pi, np.pi) == -np.pi
    assert model.discretize_angular_velocity(-8) == 0
    assert model.discretize_angular_velocity(8) == model.angular_velocity_bins - 1
    assert model.discretize_angular_velocity(-100) == 0
    assert model.discretize_angular_velocity(100) == model.angular_velocity_bins - 1

    for state in range(model.state_space):
        assert model.state_to_index(*model.index_to_state(state)) == state

    with pytest.raises(ValueError, match="angular_velocity_idx"):
        model.state_to_index(1, -1)
    with pytest.raises(ValueError, match="index"):
        model.index_to_state(model.state_space)
    with pytest.raises(ValueError, match="action"):
        model.get_action_value(-1)
    with pytest.raises(ValueError, match="action"):
        model.get_action_value(model.action_space)
    with pytest.raises(ValueError, match="finite"):
        model.discretize_angle(np.nan)
    with pytest.raises(ValueError, match="three finite"):
        model.transform_cont_obs([1, 0, np.inf])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim_samples": 2},
        {"n_workers": 0},
    ],
)
def test_pendulum_rejects_configuration_that_cannot_build_a_model(tmp_path, kwargs):
    options = {
        "angle_bins": 3,
        "angular_velocity_bins": 3,
        "torque_bins": 3,
        "n_workers": 1,
        "cache_dir": tmp_path,
        "dim_samples": 3,
    }
    options.update(kwargs)

    with pytest.raises(ValueError):
        DiscretizedPendulum(**options)


def test_pendulum_midpoint_transitions_match_current_gymnasium(tmp_path):
    env = PendulumWrapper(
        gym.make("Pendulum-v1"),
        angle_bins=3,
        angular_velocity_bins=3,
        torque_bins=3,
        n_workers=1,
        cache_dir=tmp_path,
        dim_samples=3,
    )
    try:
        model = env.discretized_pendulum
        env.reset(seed=417)
        for state in range(model.state_space):
            angle, angular_velocity = index_to_continous_state(
                state,
                model.angle_bin_edges,
                model.angular_velocity_bin_edges,
            )
            for action in range(model.action_space):
                env.unwrapped.state = np.array([angle, angular_velocity])
                observation, reward, terminated, truncated, _ = env.step(action)

                assert model.P[state][action] == [
                    (1.0, observation, pytest.approx(reward), terminated)
                ]
                assert not truncated
                assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_rendering_uses_a_supplied_human_mode_wrapper_without_closing_it():
    env_id = "BetterMDPTools-RenderTracking-v0"
    register_test_env(env_id, RenderTrackingEnv)
    RenderTrackingEnv.instances = []
    env = OffsetObservation(gym.make(env_id, render_mode="human"))
    try:
        scores = TestEnv.test_env(env, render=True, n_iters=1, pi={1: 0})

        np.testing.assert_array_equal(scores, [1.0])
        assert RenderTrackingEnv.instances[-1].render_calls == 1
        assert not RenderTrackingEnv.instances[-1].closed
        assert len(RenderTrackingEnv.instances) == 1
    finally:
        env.close()

    assert RenderTrackingEnv.instances[-1].closed


def test_rendering_rejects_an_unreproducible_non_rendering_wrapper():
    env_id = "BetterMDPTools-RenderTracking-v0"
    register_test_env(env_id, RenderTrackingEnv)
    RenderTrackingEnv.instances = []
    env = OffsetObservation(gym.make(env_id))
    try:
        with pytest.raises(ValueError, match="cannot re-create this wrapper stack"):
            TestEnv.test_env(env, render=True, n_iters=1, pi={1: 0})

        assert len(RenderTrackingEnv.instances) == 1
        assert not RenderTrackingEnv.instances[-1].closed
    finally:
        env.close()


def test_rendering_closes_an_internally_recreated_environment_after_failure():
    env_id = "BetterMDPTools-RenderTracking-v0"
    register_test_env(env_id, RenderTrackingEnv)
    RenderTrackingEnv.instances = []
    env = gym.make(env_id)
    try:
        with pytest.raises(KeyError):
            TestEnv.test_env(env, render=True, n_iters=1, pi={})

        assert len(RenderTrackingEnv.instances) == 2
        assert RenderTrackingEnv.instances[-1].closed
        assert not RenderTrackingEnv.instances[0].closed
    finally:
        env.close()


def test_experiment_run_closes_its_environment_on_success_and_failure():
    env_id = "BetterMDPTools-Lifecycle-v0"
    register_test_env(env_id, LifecycleEnv)
    LifecycleEnv.instances = []

    run(algo="q_learning", env_id=env_id, algo_kwargs={"n_episodes": 2})
    assert LifecycleEnv.instances[-1].closed

    with pytest.raises(ValueError, match="Unknown algorithm"):
        run(algo="unknown", env_id=env_id)
    assert LifecycleEnv.instances[-1].closed


def test_environment_factory_closes_an_environment_when_adaptation_fails():
    env_id = "BetterMDPTools-ModelLessLifecycle-v0"
    register_test_env(env_id, ModelLessLifecycleEnv)
    ModelLessLifecycleEnv.instances = []

    with pytest.raises(ValueError) as error:
        EnvFactory().make(env_id)

    message = str(error.value)
    assert (
        "native-P route is unavailable because it does not expose native P" in message
    )
    assert "wrapper-P route is unavailable" in message
    assert "no explicit or registered bettermdptools wrapper was found" in message
    assert ModelLessLifecycleEnv.instances[-1].closed


def test_blackjack_policy_map_excludes_the_terminal_sink():
    env = BlackjackWrapper(gym.make("Blackjack-v1"))
    try:
        values, _, policy = Planner(env.P).value_iteration()
        decision_values = values[:-1]
        decision_policy = {
            state: policy[state] for state in range(len(decision_values))
        }

        mapped_values, mapped_policy = Plots.get_policy_map(
            decision_policy,
            decision_values,
            {0: "S", 1: "H"},
            (29, 10),
        )

        assert mapped_values.shape == (29, 10)
        assert mapped_policy.shape == (29, 10)
    finally:
        env.close()
