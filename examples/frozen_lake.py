# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.rl import QLearner as QL
from bettermdptoolbox.planning import ValueIteration as VI
from bettermdptoolbox.planning import PolicyIteration as PI
from test_env import TestEnv


class FrozenLake:
    def __init__(self):
        self.env = gym.make('FrozenLake8x8-v1')


if __name__ == "__main__":

    frozen_lake = FrozenLake()

    # VI/PI
    # V, pi = VI(frozen_lake.env.P).value_iteration()
    # V, pi = PI(frozen_lake.env.P).policy_iteration()

    # Q-learning
    QL = QL(frozen_lake.env)
    Q, V, pi, Q_track, pi_track = QL.q_learning()

    test_scores = TestEnv.test_env(env=frozen_lake.env, user_input=False, pi=pi)
