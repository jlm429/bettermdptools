# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import gym
import pygame
from bettermdptoolbox.rl import QLearner as QL
from bettermdptoolbox.rl import SARSA as SARSA
import numpy as np
from bettermdptoolbox.planning import ValueIteration as VI
from bettermdptoolbox.planning import PolicyIteration as PI
from test_env import TestEnv
import pickle


class Blackjack:
    def __init__(self):
        self._env = gym.make('Blackjack-v1')
        self._convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        self._P = pickle.load(open("blackjack-envP", "rb"))
        self._n_actions = self.env.action_space.n
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
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs


if __name__ == "__main__":
    blackjack = Blackjack()

    # VI/PI
    # V, pi = VI(blackjack.P).value_iteration()
    # V, pi = PI(blackjack.P).policy_iteration()

    # Q-learning
    QL = QL(blackjack.env)
    Q, V, pi, Q_track, pi_track = QL.q_learning(blackjack.n_states, blackjack.n_actions, blackjack.convert_state_obs)

    test_scores = TestEnv.test_env(env=blackjack.env, pi=pi, user_input=False,
                                   convert_state_obs=blackjack.convert_state_obs)
