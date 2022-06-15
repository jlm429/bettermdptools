# -*- coding: utf-8 -*-

import gym
import pygame
from bettermdptoolbox.rl import QLearner as QL
from bettermdptoolbox.planning import Value_Iteration as VI
from bettermdptoolbox.planning import Policy_Iteration as PI

env = gym.make('FrozenLake8x8-v1')

#Q-learning
QL = QL(env)
nS, nA = env.observation_space.n, env.action_space.n
Q, V, pi, Q_track, pi_track = QL.q_learning(nS, nA)

#V, pi = VI().value_iteration(env.P)
#V, pi = PI().policy_iteration(env.P)


