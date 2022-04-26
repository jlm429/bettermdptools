# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import gym
import pygame
from bettermdptoolbox.RL import QLearner as QL
import numpy as np
from bettermdptoolbox.Planning import Value_Iteration as VI
from bettermdptoolbox.Planning import Policy_Iteration as PI
import pickle

class Blackjack():
    def __init__(self):
        self.env = gym.make('Blackjack-v1')
        pass

    def create_transition_matrix(self):
        #Transition probability matrix:
        #https://github.com/rhalbersma/gym-blackjack-v1
        P = pickle.load( open( "blackjack-envP", "rb" ) )
        return P

    def convert_state(self, state):
        s = 0
        if state[0] == 0: s = state[1] - 1
        else:
            if state[2]:
                s = state[0] + 6
            else:
                s = state[0] - 4
            s = int(f"{s}{state[1] - 1}")
        return s

    def test_blackjack(self, pi, n_iters):
        testScores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            done = False
            state = self.env.reset()
            state = self.convert_state(state)
            totalReward = 0
            while not done:
                self.env.render()
                action = pi(state)
                #action=np.random.randint(0,2)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.convert_state(next_state)
                state = next_state
                totalReward = reward + totalReward
            testScores[i] = totalReward
        self.env.close()
        print(testScores)
        print(testScores.sum())

if __name__=="__main__":
    blackjack = Blackjack()
    P=blackjack.create_transition_matrix()
    V, pi = VI().value_iteration(P)
    #Q, V, pi, Q_track, pi_track = QL().q_learning(blackjack.env)
    blackjack.test_blackjack(pi, 100)