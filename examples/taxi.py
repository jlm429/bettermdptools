# -*- coding: utf-8 -*-

import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv


class Taxi:
    def __init__(self):
        self.env = gym.make('Taxi-v3', render_mode=None)


if __name__ == "__main__":

    taxi = Taxi()

    # VI/PI
    # V, V_track, pi = Planner(taxi.env.P).value_iteration()
    # V, V_track, pi = Planner(taxi.env.P).policy_iteration()

    # Q-learning
    Q, V, pi, Q_track, pi_track = RL(taxi.env).q_learning()

    test_scores = TestEnv.test_env(env=taxi.env, render=True, user_input=False, pi=pi)
