# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import gym
import pygame
import numpy as np
from decorators.decorators import debug


class TestEnv:
    def __init__(self):
        pass

    @staticmethod
    @debug
    def test_env(env, n_iters=10, pi=None, convert_state_obs=lambda state, done: state):
        n_actions = env.action_space.n
        test_scores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, done = env.reset(), False
            state = convert_state_obs(state, done)
            total_reward = 0
            while not done:
                env.render()
                action = pi(state)
                if pi is None:
                    action = np.random.randint(0, n_actions)
                next_state, reward, done, info = env.step(action)
                next_state = convert_state_obs(next_state, done)
                state = next_state
                total_reward = reward + total_reward
            test_scores[i] = total_reward
        env.close()
        return test_scores
