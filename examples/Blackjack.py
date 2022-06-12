# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

#todo
#pass lambda to RL
#fix pep8 issues
#add callbacks
#update readme

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

    def create_transition_matrix(self):
        #Transition probability matrix:
        #https://github.com/rhalbersma/gym-blackjack-v1
        P = pickle.load( open( "blackjack-envP", "rb" ) )
        return P

    def test_blackjack(self, pi, n_iters):
        testScores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, done = self.env.reset(), False
            state = self.convert_state_obs(state, done)
            totalReward = 0
            while not done:
                self.env.render()
                action = pi(state)
                #action=np.random.randint(0,2)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.convert_state_obs(next_state, done)
                state = next_state
                totalReward = reward + totalReward
            testScores[i] = totalReward
        self.env.close()
        print(testScores)
        print(testScores.sum())

if __name__=="__main__":
    blackjack = Blackjack()

    #VI
    #P=blackjack.create_transition_matrix()
    #V, pi = VI().value_iteration(P)

    #Q-learning
    QL = QL(blackjack.env)
    nS, nA = 290, blackjack.env.action_space.n
    Q, V, pi, Q_track, pi_track = QL.q_learning(nS, nA, blackjack.convert_state_obs)
    blackjack.test_blackjack(pi, 100)
