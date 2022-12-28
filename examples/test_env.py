# -*- coding: utf-8 -*-
"""
Author: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Simulation of the agent's decision process after it has learned a policy.
"""

import gym
import pygame
import numpy as np


class TestEnv:
    def __init__(self):
        pass

    @staticmethod
    def test_env(env, render=True, n_iters=10, pi=None, user_input=False, convert_state_obs=lambda state, done: state):
        """
        Parameters
        ----------------------------
        env {OpenAI Gym Environment}:
            MDP problem

        render {Boolean}:
            openAI human render mode
        
        n_iters {int}, default = 10:
            Number of iterations to simulate the agent for
        
        pi {lambda}:
            Policy used to calculate action value at a given state
        
        user_input {Boolean}:
            Prompt for letting user decide which action to take at a given state
        
        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.

        
        Returns
        ----------------------------
        test_scores {list}:
            Log of reward at the end of each iteration
        """
        if render:
            # unwrap env and and reinit in 'human' render_mode
            env_name = env.unwrapped.spec.id
            env = gym.make(env_name, render_mode='human')
        n_actions = env.action_space.n
        test_scores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, info = env.reset()
            done = False
            state = convert_state_obs(state, done)
            total_reward = 0
            while not done:
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
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = convert_state_obs(next_state, done)
                state = next_state
                total_reward = reward + total_reward
            test_scores[i] = total_reward
        env.close()
        return test_scores
