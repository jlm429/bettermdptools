# -*- coding: utf-8 -*-

from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.test_env import TestEnv
import numpy as np
import itertools

class GridSearch:
    @staticmethod
    def q_learning_grid_search(env, gamma, epsilon_decay, iters):
        for i in itertools.product(gamma, epsilon_decay, iters):
            print("running q_learning with gamma:", i[0],  "epsilon decay:", i[1],  " iterations:", i[2])
            Q, V, pi, Q_track, pi_track = RL(env).q_learning(gamma=i[0], epsilon_decay_ratio=i[1], n_episodes=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            print("Avg. episode reward: ", np.mean(episode_rewards))
            print("###################")

    @staticmethod
    def sarsa_grid_search(env, gamma, epsilon_decay, iters):
        for i in itertools.product(gamma, epsilon_decay, iters):
            print("running sarsa with gamma:", i[0],  "epsilon decay:", i[1],  " iterations:", i[2])
            Q, V, pi, Q_track, pi_track = RL(env).sarsa(gamma=i[0], epsilon_decay_ratio=i[1], n_episodes=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            print("Avg. episode reward: ", np.mean(episode_rewards))
            print("###################")

    @staticmethod
    def pi_grid_search(env, gamma, n_iters, theta):
        for i in itertools.product(gamma, n_iters, theta):
            print("running PI with gamma:", i[0],  " n_iters:", i[1], " theta:", i[2])
            V, V_track, pi = Planner(env.P).policy_iteration(gamma=i[0], n_iters=i[1], theta=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            print("Avg. episode reward: ", np.mean(episode_rewards))
            print("###################")

    @staticmethod
    def vi_grid_search(env, gamma, n_iters, theta):
        for i in itertools.product(gamma, n_iters, theta):
            print("running VI with gamma:", i[0],  " n_iters:", i[1], " theta:", i[2])
            V, V_track, pi = Planner(env.P).value_iteration(gamma=i[0], n_iters=i[1], theta=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            print("Avg. episode reward: ", np.mean(episode_rewards))
            print("###################")
            