# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.rl import QLearner as QL
from bettermdptoolbox.planning import Value_Iteration as VI
from bettermdptoolbox.planning import Policy_Iteration as PI

env = gym.make('Taxi-v3')

#Q-learning
QL = QL(env)
Q, V, pi, Q_track, pi_track = QL.q_learning()

#V, pi = VI().value_iteration(env.P)
#V, pi = PI().policy_iteration(env.P)

