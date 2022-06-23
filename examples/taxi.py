# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.rl import QLearner as QL
from bettermdptoolbox.planning import Value_Iteration as VI
from bettermdptoolbox.planning import Policy_Iteration as PI
from test_env import TestEnv


class Taxi:
    def __init__(self):
        self.env = gym.make('Taxi-v3')


if __name__ == "__main__":

    taxi = Taxi()

    #Q-learning
    QL = QL(taxi.env)
    Q, V, pi, Q_track, pi_track = QL.q_learning()

    #V, pi = VI().value_iteration(env.P)
    #V, pi = PI().policy_iteration(env.P)

    test_scores = TestEnv.test_env(taxi.env, 10, pi)
    print(test_scores)
