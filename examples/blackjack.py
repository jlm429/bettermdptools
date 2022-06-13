# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

# todo
# update readme
# add callbacks

import gym
import pygame
from bettermdptoolbox.RL import QLearner as QL
from bettermdptoolbox.RL import SARSA as SARSA
import numpy as np
from bettermdptoolbox.Planning import Value_Iteration as VI
from bettermdptoolbox.Planning import Policy_Iteration as PI
import pickle


class Blackjack():
    def __init__(self):
        self.env = gym.make('Blackjack-v1')
        self.convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))

    @staticmethod
    def create_transition_matrix():
        # Transition probability matrix:
        # https://github.com/rhalbersma/gym-blackjack-v1
        p = pickle.load(open("blackjack-envP", "rb"))
        return p

    def test_blackjack(self, pi, n_iters):
        test_scores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, done = self.env.reset(), False
            state = self.convert_state_obs(state, done)
            total_reward = 0
            while not done:
                self.env.render()
                action = pi(state)
                # action=np.random.randint(0,2)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.convert_state_obs(next_state, done)
                state = next_state
                total_reward = reward + total_reward
            test_scores[i] = total_reward
        self.env.close()
        print(test_scores)
        print(test_scores.sum())


if __name__ == "__main__":
    blackjack = Blackjack()
    p = blackjack.create_transition_matrix()
    n_states = len(p)
    n_actions = blackjack.env.action_space.n

    # VI
    # V, pi = VI().value_iteration(p)

    # Q-learning
    QL = QL(blackjack.env)
    Q, V, pi, Q_track, pi_track = QL.q_learning(n_states, n_actions, blackjack.convert_state_obs)
    blackjack.test_blackjack(pi, 100)