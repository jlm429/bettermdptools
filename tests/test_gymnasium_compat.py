import warnings
from numbers import Integral, Real

import gymnasium as gym
import numpy as np
import pytest

from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.envs.acrobot_model import DiscretizedAcrobot
from bettermdptools.envs.acrobot_wrapper import AcrobotWrapper
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_model import DiscretizedCartPole
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
                assert np.isfinite(probability)
                assert 0.0 <= probability <= 1.0
                assert isinstance(next_state, Integral)
                assert 0 <= next_state < n_states
                assert isinstance(reward, Real)
                assert np.isfinite(reward)
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


def test_blackjack_state_abstraction_covers_every_reachable_observation():
    env = BlackjackWrapper(gym.make("Blackjack-v1"))
    try:
        hard_hands = {
            (player_sum, dealer_card, 0)
            for player_sum in range(4, 22)
            for dealer_card in range(1, 11)
        }
        soft_hands = {
            (player_sum, dealer_card, 1)
            for player_sum in range(12, 22)
            for dealer_card in range(1, 11)
        }
        bust_hands = {
            (player_sum, dealer_card, 0)
            for player_sum in range(22, 32)
            for dealer_card in range(1, 11)
        }

        ordinary_decision_states = {
            env.transform_obs(observation) for observation in hard_hands | soft_hands
        }
        natural_states = {
            env.transform_obs((21, dealer_card, 1), natural=True)
            for dealer_card in range(1, 11)
        }
        bust_states = {env.transform_obs(observation) for observation in bust_hands}

        assert len(hard_hands | soft_hands) == 280
        assert len(bust_hands) == 100
        assert ordinary_decision_states == set(range(280))
        assert natural_states == set(range(280, 290))
        assert bust_states == {290}
        assert env.observation_space == gym.spaces.Discrete(291)

        assert env.transform_obs((21, 2, 0)) == 170
        assert env.transform_obs((21, 2, 1)) == 270
        assert env.transform_obs((21, 2, 1), natural=True) == 280
        assert_transition_model(env.P, n_states=291, n_actions=2)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_kwargs", "expected_rewards"),
    [
        ({"natural": False, "sab": False}, {0.0, 1.0}),
        ({"natural": True, "sab": False}, {0.0, 1.5}),
        ({"sab": True}, {1.0}),
    ],
)
def test_blackjack_model_preserves_configured_natural_rewards(
    env_kwargs, expected_rewards
):
    base_env = gym.make("Blackjack-v1", **env_kwargs)
    env = BlackjackWrapper(base_env)
    try:
        natural_state = env.transform_obs((21, 2, 1), natural=True)
        assert {
            reward for _, _, reward, _ in env.P[natural_state][0]
        } == expected_rewards
    finally:
        env.close()


def test_blackjack_model_matches_hit_stick_and_bust_rules():
    env = BlackjackWrapper(gym.make("Blackjack-v1"))
    try:
        dealer_card = 10
        hard_20 = env.transform_obs((20, dealer_card, 0))
        hard_21 = env.transform_obs((21, dealer_card, 0))
        soft_21 = env.transform_obs((21, dealer_card, 1))

        assert env.P[hard_20][1] == [
            (1 / 13, hard_21, 0.0, False),
            (12 / 13, 290, -1.0, True),
        ]
        assert [
            (probability, reward, terminal)
            for probability, _, reward, terminal in env.P[hard_21][0]
        ] == [
            (probability, reward, terminal)
            for probability, _, reward, terminal in env.P[soft_21][0]
        ]
        assert {reward for _, _, reward, _ in env.P[soft_21][0]} == {0.0, 1.0}
        assert all(not terminal for _, _, _, terminal in env.P[soft_21][1])
        assert {next_state for _, next_state, _, _ in env.P[soft_21][1]} == {
            env.transform_obs((player_sum, dealer_card, 0))
            for player_sum in range(12, 22)
        }
    finally:
        env.close()


def test_blackjack_reset_context_distinguishes_natural_from_soft_21():
    env = BlackjackWrapper(gym.make("Blackjack-v1"))
    try:
        for seed in range(1000):
            state, _ = env.reset(seed=seed)
            raw_observation = env.unwrapped._get_obs()
            if raw_observation[0] == 21 and raw_observation[2]:
                break
        else:
            pytest.fail("No seeded natural blackjack found")

        assert state == env.transform_obs(raw_observation, natural=True)
        assert state != env.transform_obs(raw_observation)

        next_state, _, terminated, truncated, _ = env.step(1)
        next_observation = env.unwrapped._get_obs()
        assert next_state == env.transform_obs(next_observation)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_acrobot_wrapper_decodes_trigonometric_observation_coordinates():
    base_env = gym.make("Acrobot-v1")
    raw_observation, _ = base_env.reset(seed=417)
    base_env.close()

    env = AcrobotWrapper(gym.make("Acrobot-v1"), angle_bins=5, velocity_bins=5)
    try:
        observation, _ = env.reset(seed=417)
        coordinates = (
            np.arctan2(raw_observation[1], raw_observation[0]),
            np.arctan2(raw_observation[3], raw_observation[2]),
            raw_observation[4],
            raw_observation[5],
        )
        grids = (
            np.linspace(-np.pi, np.pi, 5),
            np.linspace(-np.pi, np.pi, 5),
            np.linspace(-4 * np.pi, 4 * np.pi, 5),
            np.linspace(-9 * np.pi, 9 * np.pi, 5),
        )
        indices = tuple(
            np.clip(np.digitize(value, grid) - 1, 0, 4)
            for value, grid in zip(coordinates, grids)
        )
        expected = np.ravel_multi_index(indices, (5, 5, 5, 5))

        assert observation == expected
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_acrobot_model_defaults_are_finite_and_resolution_covers_full_ranges():
    default_model = DiscretizedAcrobot(precomputed_P={})
    assert (
        default_model.angle_1_bins,
        default_model.angle_2_bins,
        default_model.angular_vel_1_bins,
        default_model.angular_vel_2_bins,
    ) == (10, 10, 10, 10)
    assert default_model.n_states == 10_000

    resolution_model = DiscretizedAcrobot(
        angular_resolution_rad=np.pi,
        angular_vel_resolution_rad_per_sec=100.0,
        precomputed_P={},
    )
    assert (
        resolution_model.angle_1_bins,
        resolution_model.angle_2_bins,
        resolution_model.angular_vel_1_bins,
        resolution_model.angular_vel_2_bins,
    ) == (3, 3, 2, 2)


def test_acrobot_transform_clips_exact_boundaries_and_velocity_overflow():
    env = AcrobotWrapper(gym.make("Acrobot-v1"), angle_bins=5, velocity_bins=5)
    try:
        lower = np.array([-1.0, -0.0, -1.0, -0.0, -100.0, -100.0])
        upper = np.array([-1.0, 0.0, -1.0, 0.0, 100.0, 100.0])

        assert env.transform_obs(lower) == 0
        assert env.transform_obs(upper) == env.observation_space.n - 1
    finally:
        env.close()


def test_acrobot_model_one_step_agrees_with_seeded_gymnasium_dynamics():
    env = AcrobotWrapper(gym.make("Acrobot-v1"), angle_bins=5, velocity_bins=5)
    try:
        env.reset(seed=417)
        latent_state = np.zeros(4)
        env.unwrapped.state = latent_state.copy()
        raw_observation = env.unwrapped._get_ob()
        state = env.transform_obs(raw_observation)
        expected = env.P[state][2]

        observation, reward, terminated, truncated, _ = env.step(2)

        assert expected == [(1, observation, reward, terminated)]
        assert not truncated
    finally:
        env.close()


def test_cartpole_model_uses_current_default_rewards_and_strict_boundaries():
    model = DiscretizedCartPole(
        position_bins=3,
        velocity_bins=3,
        angular_velocity_bins=3,
        threshold_bins=0.0,
        angular_center_resolution=0.1,
        angular_outer_resolution=0.5,
    )

    assert {
        reward
        for actions in model.P.values()
        for transitions in actions.values()
        for _, _, reward, _ in transitions
    } == {1.0}

    positive_threshold_index = len(model.angle_bins) - 1
    _, reward, terminated = model.compute_next_state(
        position_idx=1,
        velocity_idx=1,
        angle_idx=positive_threshold_index,
        angular_velocity_idx=1,
        action=0,
    )
    assert reward == 1.0
    assert not terminated


def test_cartpole_transform_clips_supported_terminal_observations():
    model = DiscretizedCartPole(
        position_bins=3,
        velocity_bins=3,
        angular_velocity_bins=3,
        threshold_bins=0.02,
        angular_center_resolution=0.1,
        angular_outer_resolution=0.5,
    )

    lower = np.array([-4.8, -np.inf, -0.42, -np.inf])
    upper = np.array([4.8, np.inf, 0.42, np.inf])
    assert model.transform_obs(lower) == 0
    assert model.transform_obs(upper) == model.n_states - 1


def test_cartpole_sutton_barto_reward_option_is_preserved_in_model():
    env = CartpoleWrapper(
        gym.make("CartPole-v1", sutton_barto_reward=True),
        position_bins=3,
        velocity_bins=3,
        angular_velocity_bins=3,
        threshold_bins=0.02,
    )
    try:
        rewards_by_terminal = {
            terminal: reward
            for actions in env.P.values()
            for transitions in actions.values()
            for _, _, reward, terminal in transitions
        }
        assert rewards_by_terminal == {False: 0.0, True: -1.0}
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


def test_environment_factory_builds_the_practical_acrobot_default_model():
    bundle = EnvFactory().make("Acrobot-v1")
    try:
        observation, info = bundle.env.reset(seed=417)

        assert bundle.meta["wrapper"] == "AcrobotWrapper"
        assert bundle.nS == 10_000
        assert bundle.nA == 3
        assert len(bundle.P) == bundle.nS
        assert bundle.env.observation_space.contains(observation)
        assert isinstance(info, dict)
    finally:
        bundle.env.close()


class SeededRunEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(2)
    P = {
        0: {
            0: [(1.0, 0, 0.0, True)],
            1: [(1.0, 0, 0.0, True)],
        }
    }
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


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_run_seed_reproduces_supported_blackjack_workflow(algorithm):
    run_kwargs = {
        "algo": algorithm,
        "env_id": "Blackjack-v1",
        "seed": 417,
        "algo_kwargs": {"n_episodes": 40},
        "eval_kwargs": {"n_iters": 12},
    }

    first = run(**run_kwargs)
    second = run(**run_kwargs)

    np.testing.assert_array_equal(first.train["Q"], second.train["Q"])
    np.testing.assert_array_equal(first.train["rewards"], second.train["rewards"])
    np.testing.assert_array_equal(first.eval["scores"], second.eval["scores"])


def test_run_seed_reproduces_planning_and_evaluation():
    env_id = "BetterMDPTools-SeededRun-v0"
    if env_id not in gym.registry:
        gym.register(env_id, entry_point=SeededRunEnv)
    SeededRunEnv.instances = []

    run_kwargs = {
        "algo": "pi",
        "env_id": env_id,
        "seed": 417,
        "algo_kwargs": {"n_iters": 3},
        "eval_kwargs": {"n_iters": 4},
    }
    first = run(**run_kwargs)
    second = run(**run_kwargs)

    np.testing.assert_array_equal(first.train["V"], second.train["V"])
    assert first.train["pi"] == second.train["pi"]
    np.testing.assert_array_equal(first.eval["scores"], second.eval["scores"])
    assert [env.reset_seeds for env in SeededRunEnv.instances] == [
        [417, None, None, None],
        [417, None, None, None],
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
