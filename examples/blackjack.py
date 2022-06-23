# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import gym
import pygame
from bettermdptoolbox.rl import QLearner as QL
from bettermdptoolbox.rl import SARSA as SARSA
import numpy as np
from bettermdptoolbox.planning import Value_Iteration as VI
from bettermdptoolbox.planning import Policy_Iteration as PI
from test_env import TestEnv
import pickle


class Blackjack:
    def __init__(self):
        self.env = gym.make('Blackjack-v1')
        self.convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))

    @staticmethod
    def create_transition_matrix():
        # Transition probability matrix:
        # https://github.com/rhalbersma/gym-blackjack-v1
        P = pickle.load(open("blackjack-envP", "rb"))
        return P


if __name__ == "__main__":
    blackjack = Blackjack()
    P = blackjack.create_transition_matrix()
    n_states = len(P)
    n_actions = blackjack.env.action_space.n

    # VI
    # V, pi = VI().value_iteration(P)

    # Q-learning
    QL = QL(blackjack.env)
    Q, V, pi, Q_track, pi_track = QL.q_learning(n_states, n_actions, blackjack.convert_state_obs)

    test_scores = TestEnv.test_env(blackjack.env, 10, pi, blackjack.convert_state_obs)
