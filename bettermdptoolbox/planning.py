"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Model-based learning algorithms: Value Iteration and Policy Iteration

Assumes prior knowledge of the type of reward available to the agent
for iterating to an optimal policy and reward value for a given MDP.
"""

import numpy as np
from decorators.decorators import print_runtime


class Planning:
    def __init__(self, P):
        self.P = P


class ValueIteration(Planning):
    def __init__(self, P):
        Planning.__init__(self, P)

    @print_runtime
    def value_iteration(self, gamma=1.0, theta=1e-10):
        """
        Parameters
        ----------------------------
        gamma {float}: discount factor
        
        theta {float}: Convergence criteria for checking convergence to optimal
        
        
        Returns
        ----------------------------
        V {array-like}, shape(len(self.P)):
            Optimal value array
            
        pi {lambda}, input state value, output action value:
            Optimal policy which maps state action value
        """
        
        V = np.zeros(len(self.P), dtype=np.float64)
        while True:
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                break
            V = np.max(Q, axis=1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return V, pi


class PolicyIteration(Planning):
    def __init__(self, P):
        Planning.__init__(self, P)

    @print_runtime
    def policy_iteration(self, gamma=1.0, theta=1e-10):
        """
        Parameters
        ----------------------------
        gamma {float}: discount factor
        
        theta {float}: Convergence criteria for checking convergence to optimal
        
        
        Returns
        ----------------------------
        V {array-like}, shape(len(self.P)):
            Optimal value array
            
        pi {lambda}, input state value, output action value:
            Optimal policy which maps state action value
        """

        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))
        pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        while True:
            old_pi = {s: pi(s) for s in range(len(self.P))}
            V = self.policy_evaluation(pi, V, gamma, theta)
            pi = self.policy_improvement(V, gamma)
            if old_pi == {s: pi(s) for s in range(len(self.P))}:
                break
        return V, pi

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi(s)]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi
