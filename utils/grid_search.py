# -*- coding: utf-8 -*-

from algorithms.rl import RL
from algorithms.planner import Planner
from utils.test_env import TestEnv
import numpy as np
import itertools

class GridSearch:
    @staticmethod
    def Q_learning_grid_search(env, gamma, epsilon_decay, iters):
        for i in itertools.product(gamma, epsilon_decay, iters):
            print("running Q_learning with gamma:", i[0],  "epsilon decay:", i[1],  " iterations:", i[2])
            Q, V, pi, Q_track, pi_track = RL(env).q_learning(gamma=i[0], epsilon_decay_ratio=i[1], n_episodes=i[2])
            episode_rewards = TestEnv.test_env(env=env, n_iters=100, render=False, user_input=False, pi=pi)
            print("Avg. episode reward: ", np.mean(episode_rewards))
            print("###################")

    @staticmethod
    def planner_grid_search(env, gamma, n_iters, theta):
        for i in itertools.product(gamma, n_iters, theta):
            print("running planner with gamma:", i[0],  " n_iters:", i[1], " theta:", i[2])
            V, V_track, pi = Planner(env.P).policy_iteration(gamma=i[0], n_iters=i[1], theta=i[2])
            episode_rewards = TestEnv.test_env(env, n_iters=100, render=False, user_input=False, pi=pi)
            print("Avg. episode reward: ", np.mean(episode_rewards))
            print("###################")