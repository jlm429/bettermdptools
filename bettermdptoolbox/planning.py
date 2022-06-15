"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

import numpy as np

class Planning():
    def __init__(self):
        pass

class Value_Iteration(Planning):
    def __init__(self):
        pass

    def value_iteration(self,P, gamma=1.0, theta=1e-10):
        V = np.zeros(len(P), dtype=np.float64)
        while True:
            Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
            for s in range(len(P)):
                for a in range(len(P[s])):
                    for prob, next_state, reward, done in P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                break
            V = np.max(Q, axis=1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return V, pi

class Policy_Iteration(Planning):
    def __init__(self):
        pass

    def policy_iteration(self, P, gamma=1.0, theta=1e-10):
        random_actions = np.random.choice(tuple(P[0].keys()), len(P))
        pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
        while True:
            old_pi = {s: pi(s) for s in range(len(P))}
            V = self.policy_evaluation(pi, P, gamma, theta)
            pi = self.policy_improvement(V, P, gamma)
            if old_pi == {s: pi(s) for s in range(len(P))}:
                break
        return V, pi

    def policy_evaluation(self, pi, P, gamma=1.0, theta=1e-10):
        prev_V = np.zeros(len(P), dtype=np.float64)
        while True:
            V = np.zeros(len(P), dtype=np.float64)
            for s in range(len(P)):
                for prob, next_state, reward, done in P[s][pi(s)]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V, P, gamma=1.0):
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi