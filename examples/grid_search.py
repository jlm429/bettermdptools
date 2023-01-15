# -*- coding: utf-8 -*-

import gym
import pygame
from examples.test_env import TestEnv
from algorithms.rl import RL
import itertools


class GridSearch:
    @staticmethod
    def Q_learning_grid_search(env, epsilon_decay, iters):
        for i in itertools.product(epsilon_decay, iters):
            print("running -- with epsilon decay: ", i[0],  " iterations: ", i[1])
            Q, V, pi, Q_track, pi_track = RL(env).q_learning(epsilon_decay_ratio=i[0], n_episodes=i[1])


if __name__ == "__main__":
    frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
    epsilon_decay = [.4, .7, .9]
    iters = [500, 5000, 50000]
    GridSearch.Q_learning_grid_search(frozen_lake.env, epsilon_decay, iters)
