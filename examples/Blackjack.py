# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import gym
import pygame
from bettermdptoolbox.RL import QLearner as QL
import numpy as np
from bettermdptoolbox.Planning import Value_Iteration as VI

class Blackjack():
    def __init__(self):
        self.env = gym.make('Blackjack-v1')
        pass

    def create_transition_matrix(self):
        P = 0
        return P

    def test_blackjack(self):
        testScores = np.full([100], np.nan)
        for i in range(0, 100):
            done = False
            state = self.env.reset()
            state = int(f"{state[0]}{state[1]}{int(state[2])}")
            totalReward = 0
            while not done:
                self.env.render()
                action = pi(state)
                #action=np.random.randint(0,2)
                next_state, reward, done, blah = self.env.step(action)
                next_state = int(f"{next_state[0]}{next_state[1]}{int(next_state[2])}")
                state = next_state
                totalReward = reward + totalReward
            testScores[i] = totalReward
        self.env.close()
        print(testScores)
        print(testScores.sum())

if __name__=="__main__":
    #env = gym.make('Blackjack-v1')
    blackjack = Blackjack()
    Q, V, pi, Q_track, pi_track = QL().q_learning(blackjack.env)
    blackjack.test_blackjack()