import unittest
import gymnasium as gym
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
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

        warnings.filterwarnings('ignore')

    def test_blackjack_value_iteration(self):
        V, V_track, pi = Planner(self.blackjack.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.blackjack, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_cartpole_value_iteration(self):
        V, V_track, pi = Planner(self.cartpole.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.cartpole, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_fl_value_iteration(self):
        V, V_track, pi = Planner(self.frozen_lake.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.frozen_lake, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_taxi_value_iteration(self):
        V, V_track, pi = Planner(self.taxi.P).value_iteration(n_iters=2)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.taxi, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_blackjack_policy_iteration(self):
        V, V_track, pi = Planner(self.blackjack.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.blackjack, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_cartpole_policy_iteration(self):
        V, V_track, pi = Planner(self.cartpole.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.cartpole, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_fl_policy_iteration(self):
        V, V_track, pi = Planner(self.frozen_lake.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.frozen_lake, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_taxi_policy_iteration(self):
        V, V_track, pi = Planner(self.taxi.P).policy_iteration(n_iters=1)
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.taxi, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_blackjack_q_learning(self):
        Q, V, pi, Q_track, pi_track = RL(self.blackjack).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.blackjack, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_cartpole_q_learning(self):
        Q, V, pi, Q_track, pi_track = RL(self.cartpole).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.cartpole, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_fl_q_learning(self):
        Q, V, pi, Q_track, pi_track = RL(self.frozen_lake).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.frozen_lake, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

    def test_taxi_q_learning(self):
        Q, V, pi, Q_track, pi_track = RL(self.taxi).q_learning(n_episodes=2)
        self.assertIsNotNone(Q, "Q-table should not be None")
        self.assertIsNotNone(V, "Value function should not be None")
        self.assertIsNotNone(pi, "Policy should not be None")

        test_scores = TestEnv.test_env(env=self.taxi, n_iters=1, pi=pi)
        mean_score = np.mean(test_scores)
        self.assertIsNotNone(mean_score, "Mean test score should not be None")

if __name__ == '__main__':
    unittest.main()
