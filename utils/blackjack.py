# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import os
import gymnasium as gym
import pickle


class Blackjack:
    def __init__(self):
        self._wrapped_env = gym.make('Blackjack-v1', render_mode=None)
        # Explanation of convert_state_obs lambda:
        # Lambda function assigned to the variable `self._convert_state_obs` takes parameter, `state` and
        # converts the input into a compact single integer value by concatenating player hand with dealer card.
        # See comments below for further information.
        self._convert_state_obs = lambda state: (
        int(f"{28}{(state[1] - 2) % 10}") if (state[0]==21 and state[2])
            else int(f"{27}{(state[1] - 2) % 10}") if (state[0]==21 and not state[2])
            else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2]
            else int(f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        #    Observations:
        #   There are 29 * 10 = 290 discrete observable states:
        #    29 player hands: H4-H21, S12-S21, BJ (0-28)
        #     H4   =  0
        #     H5   =  1
        #     H6   =  2
        #     H7   =  3
        #     H8   =  4
        #     H9   =  5
        #     H10  =  6
        #     H11  =  7
        #     H12  =  8
        #     H13  =  9
        #     H14  = 10
        #     H15  = 11
        #     H16  = 12
        #     H17  = 13
        #     H18  = 14
        #     H19  = 15
        #     H20  = 16
        #     H21  = 17
        #     S12  = 18
        #     S13  = 19
        #     S14  = 20
        #     S15  = 21
        #     S16  = 22
        #     S17  = 23
        #     S18  = 24
        #     S19  = 25
        #     S20  = 26
        #     S21  = 27
        #     BJ   = 28
        #    Concatenated with 10 dealer cards: 2-9, T, A (0-9)
        #     _2 = 0
        #     _3 = 1
        #     _4 = 2
        #     _5 = 3
        #     _6 = 4
        #     _7 = 5
        #     _8 = 6
        #     _9 = 7
        #     _T = 8 # 10, J, Q, K are all denoted as T
        #     _A = 9
        current_dir = os.path.dirname(__file__)
        file_name = 'blackjack-envP.pickle'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
        self._n_actions = self._wrapped_env.action_space.n
        self._n_states = len(self._P)

    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @wrapped_env.setter
    def env(self, env):
        self._wrapped_env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs
