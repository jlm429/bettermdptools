import gzip
import pickle
import sys
import warnings
from importlib import metadata
from pathlib import Path
from types import MethodType

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.envs import pendulum_discretized
from bettermdptools.envs.acrobot_wrapper import AcrobotWrapper
from bettermdptools.envs.binning import generate_bin_edges
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from bettermdptools.envs.pendulum_discretized import (
    DiscretizedPendulum,
    angle_normalize,
    index_to_continous_state,
    index_to_continuous_state,
    wrap,
)
from bettermdptools.envs.pendulum_wrapper import PendulumWrapper
from bettermdptools.experiments import run
from bettermdptools.experiments.env_factory import EnvFactory
from bettermdptools.plotting import prepare_policy_grid
from bettermdptools.utils.test_env import TestEnv


@pytest.mark.parametrize(
    "env_id",
    (
        "Blackjack-v1",
        "CartPole-v1",
        "Acrobot-v1",
        "Pendulum-v1",
        "FrozenLake-v1",
        "Taxi-v4",
    ),
)
def test_supported_gymnasium_environments_render_rgb_arrays(env_id):
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        env.reset(seed=417)
        frame = env.render()

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_rendering_without_pygame_ce_reports_the_supported_install(monkeypatch):
    real_version = metadata.version

    def version(distribution_name):
        if distribution_name == "pygame-ce":
            raise metadata.PackageNotFoundError(distribution_name)
        return real_version(distribution_name)

    monkeypatch.setattr(metadata, "version", version)
    env = RenderTrackingEnv()
    try:
        with pytest.raises(
            gym.error.DependencyNotInstalled,
            match=r"bettermdptools\[rendering\].*Classic pygame is not supported",
        ):
            TestEnv.test_env(env, render=True, n_iters=1, pi={0: 0})
    finally:
        env.close()


def test_rendering_rejects_installed_classic_pygame(monkeypatch):
    versions = {"pygame-ce": "2.5.8", "pygame": "2.6.1"}
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    env = RenderTrackingEnv()
    try:
        with pytest.raises(
            gym.error.DependencyNotInstalled,
            match="Classic pygame 2.6.1 is installed.*supports only pygame-ce",
        ):
            TestEnv.test_env(env, render=True, n_iters=1, pi={0: 0})
    finally:
        env.close()


class RenderTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)
    instances = []
    events = []

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
        type(self).events.append(("render", id(self)))

    def close(self):
        self.closed = True
        type(self).events.append(("close", id(self)))


class OffsetObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(2)

    def observation(self, observation):
        return int(observation) + 1


class ArrayRenderingWithoutFpsEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.render_mode = None
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        assert self.action_space.contains(action)
        return 0, 0.0, True, False, {}

    def render(self):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


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


def track_render_lifecycle(env):
    events = []
    raw_env = env.unwrapped
    caller_id = id(raw_env)

    def render(instance):
        events.append(("render", id(instance)))

    def close(instance):
        events.append(("close", id(instance)))

    raw_env.render = MethodType(render, raw_env)
    raw_env.close = MethodType(close, raw_env)
    return events, caller_id


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.states = []

    def __getitem__(self, state):
        self.states.append(state)
        return self.action


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
    "method_name",
    ["value_iteration", "value_iteration_vectorized", "policy_iteration"],
)
def test_planning_metadata_reports_exact_valid_history_without_changing_default(
    method_name,
):
    model = {0: {0: [(1.0, 0, 0.0, True)]}}
    np.random.seed(0)
    default_result = getattr(Planner(model), method_name)(n_iters=5)
    np.random.seed(0)
    V, V_track, pi, metadata = getattr(Planner(model), method_name)(
        n_iters=5, return_metadata=True
    )

    assert len(default_result) == 3
    np.testing.assert_array_equal(V, default_result[0])
    np.testing.assert_array_equal(V_track, default_result[1])
    assert pi == default_result[2]
    assert metadata.iterations == 1
    assert metadata.history_length == 2
    assert metadata.converged is True
    np.testing.assert_array_equal(V_track[: metadata.history_length], [[0.0], [0.0]])


@pytest.mark.parametrize(
    "method_name",
    ["value_iteration", "value_iteration_vectorized", "policy_iteration"],
)
def test_planning_metadata_does_not_treat_valid_zero_rows_as_padding(method_name):
    model = {0: {0: [(1.0, 0, 0.0, True)]}}
    np.random.seed(0)

    with pytest.warns(UserWarning, match="Max iterations reached"):
        _, V_track, _, metadata = getattr(Planner(model), method_name)(
            n_iters=4, theta=0.0, return_metadata=True
        )

    assert metadata.iterations == 3
    assert metadata.history_length == 4
    assert metadata.converged is False
    np.testing.assert_array_equal(V_track, np.zeros((4, 1)))


def test_experiment_planning_metadata_is_available_when_requested():
    result = run(
        algo="vi",
        env_id="FrozenLake-v1",
        env_kwargs={"is_slippery": False},
        algo_kwargs={"n_iters": 20, "return_metadata": True},
    )

    metadata = result.train["planning_metadata"]
    assert 2 <= metadata.history_length <= 20
    assert metadata.history_length == metadata.iterations + 1
    assert result.train["V_track"].shape == (20, 16)


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
    centered_edges = generate_bin_edges(
        np.float32(2), np.int64(5), np.float32(3), center=True
    )
    centered = np.asarray(centered_edges)
    outer_fine = np.asarray(generate_bin_edges(2, 5, 3, center=False))

    assert len(centered) == 6
    assert centered[0] == -2
    assert centered[-1] == 2
    assert all(type(edge) is float for edge in centered_edges)
    assert np.isfinite(centered).all()
    assert np.all(np.diff(centered) > 0)
    np.testing.assert_allclose(centered, -centered[::-1])
    assert np.diff(centered)[0] > np.diff(centered)[2]
    assert np.diff(outer_fine)[0] < np.diff(outer_fine)[2]


def test_generate_bin_edges_avoids_overflow_for_finite_float_limits():
    edges = np.asarray(generate_bin_edges(np.finfo(float).max, 3, 3))

    assert np.isfinite(edges).all()
    assert np.all(np.diff(edges) > 0)


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(float).max,
    reason="long double has no wider finite range",
)
def test_generate_bin_edges_rejects_values_that_overflow_float():
    range_limit = np.longdouble(np.finfo(float).max) * 2

    with pytest.raises(ValueError, match="representable as a finite float"):
        generate_bin_edges(range_limit, 3, 3)


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


def test_pendulum_interactive_context_falls_back_to_serial_workers(
    tmp_path, monkeypatch
):
    class UnexpectedExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("an interactive main module cannot start workers")

    monkeypatch.setattr(pendulum_discretized, "get_start_method", lambda: "spawn")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", "<stdin>")
    monkeypatch.setattr(pendulum_discretized, "ProcessPoolExecutor", UnexpectedExecutor)

    model = DiscretizedPendulum(3, 3, 3, n_workers=2, cache_dir=tmp_path, dim_samples=3)

    assert set(model.P) == set(range(model.state_space))
    assert all(
        model.P[state][action]
        for state in range(model.state_space)
        for action in range(model.action_space)
    )


def test_pendulum_worker_failure_is_not_cached_and_later_call_retries(
    tmp_path, monkeypatch
):
    real_builder = pendulum_discretized.setup_transition_probabilities_for_state

    class Future:
        def __init__(self, args):
            self.args = args

        def result(self):
            if self.args[0] == 1:
                raise RuntimeError("worker failed")
            return real_builder(self.args)

    class FailingExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, function, args):
            return Future(args)

    monkeypatch.setattr(pendulum_discretized, "get_start_method", lambda: "fork")
    monkeypatch.setattr(pendulum_discretized, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(pendulum_discretized, "as_completed", lambda futures: futures)

    failed_env = gym.make("Pendulum-v1")
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            PendulumWrapper(
                failed_env,
                angle_bins=3,
                angular_velocity_bins=3,
                torque_bins=3,
                n_workers=2,
                cache_dir=tmp_path,
                dim_samples=3,
            )
    finally:
        failed_env.close()

    assert list(tmp_path.glob("*.pkl.gz")) == []

    rebuilt = PendulumWrapper(
        gym.make("Pendulum-v1"),
        angle_bins=3,
        angular_velocity_bins=3,
        torque_bins=3,
        n_workers=1,
        cache_dir=tmp_path,
        dim_samples=3,
    )
    try:
        assert set(rebuilt.P) == set(range(rebuilt.observation_space.n))
        assert all(
            rebuilt.P[state][action]
            for state in rebuilt.P
            for action in range(rebuilt.action_space.n)
        )
    finally:
        rebuilt.close()


def test_pendulum_rebuilds_an_incomplete_cached_model(tmp_path):
    model = DiscretizedPendulum(3, 3, 3, n_workers=1, cache_dir=tmp_path, dim_samples=3)
    cache_file = next(tmp_path.glob("*.pkl.gz"))
    with gzip.open(cache_file, "wb") as file:
        pickle.dump({}, file)

    rebuilt = DiscretizedPendulum(
        3, 3, 3, n_workers=1, cache_dir=tmp_path, dim_samples=3
    )

    assert rebuilt.P == model.P


def test_pendulum_publishes_cache_atomically(tmp_path, monkeypatch):
    def fail_during_serialization(model, file):
        file.write(b"incomplete")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(pendulum_discretized.pickle, "dump", fail_during_serialization)

    with pytest.raises(RuntimeError, match="serialization failed"):
        DiscretizedPendulum(3, 3, 3, n_workers=1, cache_dir=tmp_path, dim_samples=3)

    assert list(tmp_path.iterdir()) == []


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
            angle, angular_velocity = index_to_continuous_state(
                state,
                model.angle_bin_edges,
                model.angular_velocity_bin_edges,
            )
            assert (angle, angular_velocity) == index_to_continous_state(
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


def test_rendering_preserves_a_native_p_environment_and_caller_ownership():
    env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=1)
    events, caller_id = track_render_lifecycle(env)
    try:
        policy = FixedPolicy(0)

        scores = TestEnv.test_env(
            env,
            render=True,
            n_iters=1,
            pi=policy,
            seed=417,
        )

        np.testing.assert_array_equal(scores, [0.0])
        assert policy.states == [0]
        rendered_ids = {
            instance_id for event, instance_id in events if event == "render"
        }
        assert len(rendered_ids) == 1
        rendered_id = rendered_ids.pop()
        assert rendered_id != caller_id
        assert ("close", rendered_id) in events
        assert ("close", caller_id) not in events
        observation, _ = env.reset(seed=417)
        assert env.observation_space.contains(observation)
    finally:
        env.close()

    assert ("close", caller_id) in events


@pytest.mark.parametrize(
    "env_name",
    ["blackjack", "cartpole", "acrobot", "pendulum"],
)
def test_rendering_preserves_supported_modeled_wrappers_and_closes_copy(
    env_name, tmp_path
):
    base_env = gym.make(
        {
            "blackjack": "Blackjack-v1",
            "cartpole": "CartPole-v1",
            "acrobot": "Acrobot-v1",
            "pendulum": "Pendulum-v1",
        }[env_name],
        max_episode_steps=1,
    )
    events, caller_id = track_render_lifecycle(base_env)
    if env_name == "blackjack":
        env = BlackjackWrapper(base_env)
    elif env_name == "cartpole":
        env = CartpoleWrapper(base_env, position_bins=2, velocity_bins=2)
    elif env_name == "acrobot":
        env = AcrobotWrapper(base_env, angle_bins=2, velocity_bins=2)
    else:
        env = PendulumWrapper(
            base_env,
            angle_bins=3,
            angular_velocity_bins=3,
            torque_bins=3,
            n_workers=1,
            cache_dir=tmp_path / "pendulum-cache",
            dim_samples=3,
        )

    try:
        policy = FixedPolicy(0)
        expected_action_count = env.action_space.n

        scores = TestEnv.test_env(
            env,
            render=True,
            n_iters=1,
            pi=policy,
            seed=417,
        )

        assert np.isfinite(scores).all()
        assert env.action_space.n == expected_action_count
        assert policy.states
        assert all(env.observation_space.contains(state) for state in policy.states)
        rendered_ids = {
            instance_id for event, instance_id in events if event == "render"
        }
        assert len(rendered_ids) == 1
        rendered_id = rendered_ids.pop()
        assert rendered_id != caller_id
        assert ("close", rendered_id) in events
        assert ("close", caller_id) not in events
        observation, _ = env.reset(seed=417)
        assert env.observation_space.contains(observation)
    finally:
        env.close()

    assert ("close", caller_id) in events


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


def test_rendering_preserves_an_unreproducible_wrapper_and_state_conversion():
    env_id = "BetterMDPTools-RenderTracking-v0"
    register_test_env(env_id, RenderTrackingEnv)
    RenderTrackingEnv.instances = []
    RenderTrackingEnv.events = []
    env = OffsetObservation(gym.make(env_id))
    caller_id = id(env.unwrapped)
    try:
        scores = TestEnv.test_env(
            env,
            render=True,
            n_iters=1,
            pi={11: 0},
            convert_state_obs=lambda state: state + 10,
        )

        np.testing.assert_array_equal(scores, [1.0])
        assert len(RenderTrackingEnv.instances) == 1
        assert not RenderTrackingEnv.instances[-1].closed
        rendered_ids = {
            instance_id
            for event, instance_id in RenderTrackingEnv.events
            if event == "render"
        }
        assert len(rendered_ids) == 1
        rendered_id = rendered_ids.pop()
        assert rendered_id != caller_id
        assert ("close", rendered_id) in RenderTrackingEnv.events
        assert ("close", caller_id) not in RenderTrackingEnv.events
    finally:
        env.close()


def test_rendering_closes_an_internally_copied_environment_after_failure():
    env_id = "BetterMDPTools-RenderTracking-v0"
    register_test_env(env_id, RenderTrackingEnv)
    RenderTrackingEnv.instances = []
    RenderTrackingEnv.events = []
    env = gym.make(env_id)
    caller_id = id(env.unwrapped)
    try:
        with pytest.raises(KeyError):
            TestEnv.test_env(env, render=True, n_iters=1, pi={})

        assert len(RenderTrackingEnv.instances) == 1
        assert not RenderTrackingEnv.instances[0].closed
        closed_ids = {
            instance_id
            for event, instance_id in RenderTrackingEnv.events
            if event == "close"
        }
        assert len(closed_ids) == 1
        assert caller_id not in closed_ids
    finally:
        env.close()


def test_array_only_rendering_requires_render_fps_metadata():
    env = ArrayRenderingWithoutFpsEnv()

    with pytest.raises(
        ValueError,
        match=r"requires metadata\['render_fps'\] for array-only rendering",
    ):
        TestEnv.test_env(env, render=True, n_iters=1, pi={0: 0})

    assert not env.closed


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

        policy_grid = prepare_policy_grid(
            decision_policy,
            decision_values,
            {0: "S", 1: "H"},
            (29, 10),
        )

        assert policy_grid.values.shape == (29, 10)
        assert policy_grid.actions.shape == (29, 10)
    finally:
        env.close()
