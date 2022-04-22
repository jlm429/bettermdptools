# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.RL import QLearner as QL
import numpy as np
from bettermdptoolbox.Planning import Value_Iteration as VI

def test_blackjack():
    testScores = np.full([100], np.nan)
    for i in range(0, 100):
        done = False
        state = env.reset()
        state = int(f"{state[0]}{state[1]}{int(state[2])}")
        totalReward = 0
        while not done:
            env.render()
            action = pi(state)
            #action=np.random.randint(0,2)
            next_state, reward, done, blah = env.step(action)
            next_state = int(f"{next_state[0]}{next_state[1]}{int(next_state[2])}")
            state = next_state
            totalReward = reward + totalReward
        testScores[i] = totalReward
    env.close()
    print(testScores)
    print(testScores.sum())

env = gym.make('Blackjack-v1')
Q, V, pi, Q_track, pi_track = QL().q_learning(env, n_episodes=50000, gamma=.8)
test_blackjack()


