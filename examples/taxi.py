# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.RL import QLearner as QL
from bettermdptoolbox.Planning import Value_Iteration as VI

env = gym.make('Taxi-v3')
Q, V, pi, Q_track, pi_track = QL().q_learning(env)

#V, pi = VI().value_iteration(env.P)

