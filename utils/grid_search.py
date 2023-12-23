# -*- coding: utf-8 -*-

from algorithms.rl import RL
import itertools


class GridSearch:
    @staticmethod
    def Q_learning_grid_search(env, epsilon_decay, iters):
        for i in itertools.product(epsilon_decay, iters):
            print("running -- with epsilon decay: ", i[0],  " iterations: ", i[1])
            Q, V, pi, Q_track, pi_track = RL(env).q_learning(epsilon_decay_ratio=i[0], n_episodes=i[1])
