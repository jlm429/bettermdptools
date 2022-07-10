# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import gym
import pygame
import numpy as np


class TestEnv:
    def __init__(self):
        pass

    @staticmethod
    def test_env(env, n_iters=10, pi=None, user_input=False, convert_state_obs=lambda state, done: state):
        n_actions = env.action_space.n
        test_scores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, done = env.reset(), False
            state = convert_state_obs(state, done)
            total_reward = 0
            while not done:
                env.render()
                if user_input:
                    # get user input and suggest policy output
                    print("state is %i" % state)
                    print("policy output is %i" % pi(state))
                    while True:
                        action = input("Please select 0 - %i then hit enter:\n" % int(n_actions-1))
                        try:
                            action = int(action)
                        except ValueError:
                            print("Please enter a number")
                            continue
                        if 0 <= action < n_actions:
                            break
                        else:
                            print("please enter a valid action, 0 - %i \n" % int(n_actions - 1))
                else:
                    action = pi(state)
                next_state, reward, done, info = env.step(action)
                next_state = convert_state_obs(next_state, done)
                state = next_state
                total_reward = reward + total_reward
            test_scores[i] = total_reward
        env.close()
        return test_scores
