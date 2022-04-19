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
