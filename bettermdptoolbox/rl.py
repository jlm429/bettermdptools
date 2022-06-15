"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
edited by: John Mansfield
"""

import numpy as np
from tqdm import tqdm
import gym


class RL():
    def __init__(self):
        pass

    @staticmethod
    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values


class QLearner(RL):
    def __init__(self, env):
        self.env=env

    def q_learning(self, nS, nA,
                   convert_state_obs=lambda state, done: state,
                   gamma=.99,
                   init_alpha=0.5,
                   min_alpha=0.01,
                   alpha_decay_ratio=0.5,
                   init_epsilon=1.0,
                   min_epsilon=0.1,
                   epsilon_decay_ratio=0.9,
                   n_episodes=10000):
        #nS, nA = env.observation_space.n, env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)
        for e in tqdm(range(n_episodes), leave=False):
            state, done = self.env.reset(), False
            state = convert_state_obs(state, done)
            if e % 5000 == 0:
                render=True
            while not done:
                if render==True:
                    self.env.render()
                action = select_action(state, Q, epsilons[e])
                next_state, reward, done, _ = self.env.step(action)
                next_state = convert_state_obs(next_state,done)
                td_target = reward + gamma * Q[next_state].max() * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            render=False

        V = np.max(Q, axis=1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track


class SARSA(RL):
    def __init__(self, env):
        self.env = env

    def sarsa(self, nS, nA,
              convert_state_obs=lambda state, done: state,
              gamma=.99,
              init_alpha=0.5,
              min_alpha=0.01,
              alpha_decay_ratio=0.5,
              init_epsilon=1.0,
              min_epsilon=0.1,
              epsilon_decay_ratio=0.9,
              n_episodes=10000):
        #nS, nA = env.observation_space.n, env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)

        for e in tqdm(range(n_episodes), leave=False):
            state, done = self.env.reset(), False
            state = convert_state_obs(state, done)
            action = select_action(state, Q, epsilons[e])
            if e % 5000 == 0:
                render=True
            while not done:
                if render==True:
                    self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                next_state = convert_state_obs(next_state, done)
                next_action = select_action(next_state, Q, epsilons[e])
                td_target = reward + gamma * Q[next_state][next_action] * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state, action = next_state, next_action
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            render=False

        V = np.max(Q, axis=1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track