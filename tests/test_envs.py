import unittest
import gymnasium as gym
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from bettermdptools.envs.pendulum_wrapper import PendulumWrapper
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
import numpy as np
import warnings


class TestEnvs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_env = gym.make('Blackjack-v1', render_mode=None)
        cls.blackjack = BlackjackWrapper(base_env)
        base_env = gym.make('CartPole-v1', render_mode=None)
        cls.cartpole = CartpoleWrapper(base_env)
        cls.frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
        cls.taxi = gym.make('Taxi-v3', render_mode=None)

        base_env = gym.make('Pendulum-v1', render_mode=None)
        cls.pendulum = PendulumWrapper(base_env)

        warnings.filterwarnings('ignore')

    def test_blackjack_value_iteration(self):
        V, V_track, pi, _, _ = Planner(self.blackjack.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.blackjack, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_cartpole_value_iteration(self):
        V, V_track, pi, _, _ = Planner(self.cartpole.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.cartpole, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_fl_value_iteration(self):
        V, V_track, pi, _, _ = Planner(self.frozen_lake.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.frozen_lake, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_taxi_value_iteration(self):
        V, V_track, pi, _, _ = Planner(self.taxi.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.taxi, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_pendulum_value_iteration(self):
        V, _, pi, _, _ = Planner(self.pendulum.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.pendulum, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_blackjack_policy_iteration(self):
        V, V_track, pi, _, _ = Planner(self.blackjack.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.blackjack, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_cartpole_policy_iteration(self):
        V, V_track, pi, _, _ = Planner(self.cartpole.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.cartpole, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_fl_policy_iteration(self):
        V, V_track, pi, _, _ = Planner(self.frozen_lake.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.frozen_lake, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_taxi_policy_iteration(self):
        V, V_track, pi, _, _ = Planner(self.taxi.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.taxi, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    # probably runs too long for CI
    # def test_pendulum_policy_iteration(self):
    #     V, _, pi, _, _ = Planner(self.pendulum.P).policy_iteration(n_iters=2)
    #     self.assertIsNotNone(V, "Value function should not be None")
    #     self.assertIsNotNone(pi, "Policy should not be None")

    #     test_scores = TestEnv.test_env(env=self.pendulum, n_iters=1, pi=pi)
    #     mean_score = np.mean(test_scores)
    #     self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_blackjack_q_learning(self):
        Q, V, pi, Q_track, pi_track, _, _, _, _, _ = RL(self.blackjack).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.blackjack, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_cartpole_q_learning(self):
        Q, V, pi, Q_track, pi_track, _, _, _, _, _ = RL(self.cartpole).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.cartpole, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_fl_q_learning(self):
        Q, V, pi, Q_track, pi_track, _, _, _, _, _ = RL(self.frozen_lake).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.frozen_lake, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_taxi_q_learning(self):
        Q, V, pi, Q_track, pi_track, _, _, _, _, _ = RL(self.taxi).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.taxi, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_pendulum_q_learning(self):
        Q, V, pi, Q_track, pi_track, _, _, _, _, _ = RL(self.pendulum).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.pendulum, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    # def test_fail_on_purpose(self):
    #     self.assertTrue(False, "This test should fail")

if __name__ == '__main__':
    unittest.main()
