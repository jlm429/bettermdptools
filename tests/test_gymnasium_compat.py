import warnings
from numbers import Integral, Real

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.envs.acrobot_wrapper import AcrobotWrapper
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from bettermdptools.envs.pendulum_wrapper import PendulumWrapper
from bettermdptools.experiments import run
from bettermdptools.experiments.env_factory import EnvFactory


def assert_transition_model(P, n_states, n_actions):
    assert len(P) == n_states
    for state in range(n_states):
        assert len(P[state]) == n_actions
        for action in range(n_actions):
            transitions = P[state][action]
            assert transitions
            assert np.isclose(sum(item[0] for item in transitions), 1.0)
            for probability, next_state, reward, terminal in transitions:
                assert 0.0 <= probability <= 1.0
                assert isinstance(next_state, Integral)
                assert 0 <= next_state < n_states
                assert isinstance(reward, Real)
                assert isinstance(terminal, (bool, np.bool_))


@pytest.fixture(
    params=[
        (
            "Acrobot-v1",
            AcrobotWrapper,
            {"angle_bins": 2, "velocity_bins": 2},
        ),
        (
            "CartPole-v1",
            CartpoleWrapper,
            {
                "position_bins": 2,
                "velocity_bins": 2,
                "angular_velocity_bins": 2,
            },
        ),
        (
            "Pendulum-v1",
            PendulumWrapper,
            {
                "angle_bins": 3,
                "angular_velocity_bins": 3,
                "torque_bins": 3,
                "n_workers": 1,
                "dim_samples": 3,
            },
        ),
    ],
    ids=["acrobot", "cartpole", "pendulum"],
)
def wrapped_env(request, tmp_path):
    env_id, wrapper_class, wrapper_kwargs = request.param
    if wrapper_class is PendulumWrapper:
        wrapper_kwargs = {**wrapper_kwargs, "cache_dir": str(tmp_path)}
    base_env = gym.make(env_id, max_episode_steps=3)
    env = wrapper_class(base_env, **wrapper_kwargs)
    yield env
    env.close()


def test_wrappers_follow_current_reset_step_and_space_contract(wrapped_env):
    first_observation, first_info = wrapped_env.reset(seed=417)
    second_observation, second_info = wrapped_env.reset(seed=417)

    assert first_observation == second_observation
    assert isinstance(first_info, dict)
    assert isinstance(second_info, dict)
    assert wrapped_env.observation_space.contains(first_observation)

    wrapped_env.action_space.seed(417)
    for step_number in range(1, 4):
        action = wrapped_env.action_space.sample()
        observation, reward, terminated, truncated, info = wrapped_env.step(action)
        assert wrapped_env.observation_space.contains(observation)
        assert isinstance(reward, Real)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        if terminated or truncated:
            assert step_number == 3
            assert truncated
            break
    else:
        pytest.fail("The TimeLimit truncation was not propagated by the wrapper")


def test_blackjack_wrapper_uses_current_reset_and_step_contract():
    env = BlackjackWrapper(gym.make("Blackjack-v1"))
    try:
        observation, info = env.reset(seed=417)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

        step_result = env.step(env.action_space.sample())
        assert len(step_result) == 5
        assert env.observation_space.contains(step_result[0])
        assert isinstance(step_result[2], (bool, np.bool_))
        assert isinstance(step_result[3], (bool, np.bool_))
        assert isinstance(step_result[4], dict)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_id", "wrapper_class", "wrapper_kwargs"),
    [
        ("Acrobot-v1", AcrobotWrapper, {"angle_bins": 2, "velocity_bins": 2}),
        (
            "CartPole-v1",
            CartpoleWrapper,
            {
                "position_bins": 2,
                "velocity_bins": 2,
                "angular_velocity_bins": 2,
            },
        ),
    ],
)
def test_discretized_wrapper_models_feed_planning_algorithms(
    env_id, wrapper_class, wrapper_kwargs
):
    env = wrapper_class(gym.make(env_id), **wrapper_kwargs)
    try:
        assert_transition_model(
            env.P,
            env.observation_space.n,
            env.action_space.n,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, _, policy = Planner(env.P).value_iteration(
                gamma=0.99,
                n_iters=20,
            )
        assert values.shape == (env.observation_space.n,)
        assert np.isfinite(values).all()
        assert set(policy) == set(range(env.observation_space.n))
    finally:
        env.close()


def test_pendulum_model_boundaries_and_transition_invariants(tmp_path):
    env = PendulumWrapper(
        gym.make("Pendulum-v1"),
        angle_bins=3,
        angular_velocity_bins=3,
        torque_bins=3,
        n_workers=1,
        cache_dir=str(tmp_path),
        dim_samples=3,
    )
    try:
        model = env.discretized_pendulum
        assert model.discretize_angle(-np.pi) == 0
        assert model.discretize_angle(np.pi) == model.angle_bins - 1
        assert model.discretize_angular_velocity(-8.0) == 0
        assert model.discretize_angular_velocity(8.0) == model.angular_velocity_bins - 1
        assert_transition_model(
            env.P,
            env.observation_space.n,
            env.action_space.n,
        )
        assert all(
            reward <= 0.0
            for actions in env.P.values()
            for transitions in actions.values()
            for _, _, reward, _ in transitions
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, _, policy = Planner(env.P).value_iteration(
                gamma=0.9,
                n_iters=10,
            )
        assert values.shape == (env.observation_space.n,)
        assert np.isfinite(values).all()
        assert len(policy) == env.observation_space.n
    finally:
        env.close()


def test_discrete_models_and_planning_workflow_under_gymnasium_1_3():
    for env_id in ("FrozenLake-v1", "FrozenLake8x8-v1", "Taxi-v4"):
        env = gym.make(env_id)
        try:
            assert_transition_model(
                env.unwrapped.P,
                env.observation_space.n,
                env.action_space.n,
            )
        finally:
            env.close()

    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        planner = Planner(env.unwrapped.P)
        values, _, policy = planner.value_iteration(gamma=0.99)
        vector_values, _, vector_policy = planner.value_iteration_vectorized(gamma=0.99)
        policy_values, _, policy_iteration_policy = planner.policy_iteration(gamma=0.99)

        assert np.isclose(values[0], 0.99**5)
        assert np.allclose(values, vector_values)
        assert policy == vector_policy
        assert np.allclose(values, policy_values)
        assert policy == policy_iteration_policy
    finally:
        env.close()


def test_environment_factory_reads_the_unwrapped_discrete_model():
    bundle = EnvFactory().make(
        "FrozenLake-v1",
        gym_kwargs={"is_slippery": False},
    )
    try:
        assert bundle.P is bundle.env.unwrapped.P
        assert bundle.nS == bundle.env.observation_space.n
        assert bundle.nA == bundle.env.action_space.n
    finally:
        bundle.env.close()


class SeededRunEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)
    P = {0: {0: [(1.0, 0, 0.0, True)]}}
    instances = []

    def __init__(self):
        self.reset_seeds = []
        type(self).instances.append(self)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        return 0, {}

    def step(self, action):
        assert self.action_space.contains(action)
        reward = float(self.np_random.integers(0, 2))
        return 0, reward, True, False, {}


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_run_seed_is_reproducible_end_to_end(algorithm):
    env_id = "BetterMDPTools-SeededRun-v0"
    if env_id not in gym.registry:
        gym.register(env_id, entry_point=SeededRunEnv)
    SeededRunEnv.instances = []

    run_kwargs = {
        "algo": algorithm,
        "env_id": env_id,
        "seed": 417,
        "algo_kwargs": {"n_episodes": 8},
        "eval_kwargs": {"n_iters": 4},
    }

    first = run(**run_kwargs)
    second = run(**run_kwargs)

    np.testing.assert_array_equal(first.train["Q"], second.train["Q"])
    np.testing.assert_array_equal(first.train["rewards"], second.train["rewards"])
    np.testing.assert_array_equal(first.eval["scores"], second.eval["scores"])

    expected_reset_seeds = [417, *([None] * 7), 417, *([None] * 3)]
    assert [env.reset_seeds for env in SeededRunEnv.instances] == [
        expected_reset_seeds,
        expected_reset_seeds,
    ]

    for env in SeededRunEnv.instances:
        env.close()


class AlternatingBoundaryEnv(gym.Env):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, truncate):
        self.truncate = truncate
        self.episode = -1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        observation = 1 if self.episode % 2 == 0 else 0
        return observation, {}

    def step(self, action):
        assert self.action_space.contains(action)
        if self.episode % 2 == 0:
            return 1, 10.0, True, False, {}
        return 1, 0.0, not self.truncate, self.truncate, {}


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
@pytest.mark.parametrize(
    ("truncate", "expected_value"),
    [(True, 5.0), (False, 0.0)],
    ids=["truncation-bootstraps", "termination-does-not-bootstrap"],
)
def test_td_bootstrapping_distinguishes_termination_from_truncation(
    algorithm, truncate, expected_value
):
    agent = RL(AlternatingBoundaryEnv(truncate=truncate))
    agent.select_action = lambda state, Q, epsilon: 0

    Q = getattr(agent, algorithm)(
        gamma=0.5,
        init_alpha=1.0,
        min_alpha=1.0,
        alpha_decay_ratio=0.5,
        init_epsilon=0.0,
        min_epsilon=0.0,
        epsilon_decay_ratio=0.5,
        n_episodes=4,
    )[0]

    assert Q[1, 0] == 10.0
    assert Q[0, 0] == expected_value
