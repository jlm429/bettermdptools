# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.RL import QLearner as QL
from bettermdptoolbox.Planning import Value_Iteration as VI

env = gym.make('FrozenLake8x8-v1')
Q, V, pi, Q_track, pi_track = QL().q_learning(env)

#env = gym.make('FrozenLake8x8-v1')
#V, pi = VI().value_iteration(env.P)

