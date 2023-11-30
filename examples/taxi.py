# -*- coding: utf-8 -*-

import gymnasium as gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv


class Taxi:
    def __init__(self):
        self.wrapped_env = gym.make('Taxi-v3', render_mode=None)


if __name__ == "__main__":

    taxi = Taxi()

    # VI/PI
    # V, V_track, pi = Planner(taxi.wrapped_env.P).value_iteration()
    # V, V_track, pi = Planner(taxi.wrapped_env.P).policy_iteration()

    # Q-learning
    Q, V, pi, Q_track, pi_track = RL(taxi.wrapped_env).q_learning()

    test_scores = TestEnv.test_env(env=taxi.wrapped_env, desc=None, render=True, user_input=False, pi=pi)
