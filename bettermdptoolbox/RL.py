"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

import numpy as np
from tqdm import tqdm

class RL():
    def __init__(self):
        pass

class QLearner(RL):
    def __init__(self):
        pass

    def decay_schedule(self, init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values

    def q_learning(self, env,
                   gamma=.99,
                   init_alpha=0.5,
                   min_alpha=0.01,
                   alpha_decay_ratio=0.5,
                   init_epsilon=1.0,
                   min_epsilon=0.1,
                   epsilon_decay_ratio=0.9,
                   n_episodes=30000):
        nS, nA = env.observation_space.n, env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = self.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = self.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)
        for e in tqdm(range(n_episodes), leave=False):
            state, done = env.reset(), False
            if e % 2000 == 0:
                render=True
            while not done:
                action = select_action(state, Q, epsilons[e])
                next_state, reward, done, _ = env.step(action)
                td_target = reward + gamma * Q[next_state].max() * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
                if render==True:
                    env.render()
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            render=False

        V = np.max(Q, axis=1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track