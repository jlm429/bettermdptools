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
Class that contains functions related to planning algorithms (Value Iteration, Policy Iteration). 
Planner init expects a reward and transitions matrix P, which is nested dictionary gym style discrete environment 
where P[state][action] is a list of tuples (probability, next state, reward, terminal).

Model-based learning algorithms: Value Iteration and Policy Iteration
"""

import warnings

import numpy as np


class Planner:
    def __init__(self, P):
        """
        Initializes the Planner class.

        Parameters
        ----------
        P : dict
            Transition probability matrix.
        """
        self.P = P

    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        Value Iteration algorithm.

        Parameters
        ----------
        gamma : float, optional
            Discount factor, by default 1.0.
        n_iters : int, optional
            Number of iterations, by default 1000.
        theta : float, optional
            Convergence criterion for value iteration, by default 1e-10.

        Returns
        -------
        tuple
            State values array, log of V(s) for each iteration, and policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            V = np.max(Q, axis=1)
            V_track[i] = V
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")

        pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return V, V_track, pi

    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        """
        Policy Iteration algorithm.

        Parameters
        ----------
        gamma : float, optional
            Discount factor, by default 1.0.
        n_iters : int, optional
            Number of iterations, by default 50.
        theta : float, optional
            Convergence criterion for policy evaluation, by default 1e-10.

        Returns
        -------
        tuple
            State values array, log of V(s) for each iteration, and policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            old_pi = pi
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i] = V
            pi = self.policy_improvement(V, gamma)
            if old_pi == pi:
                converged = True
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        return V, V_track, pi

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        """
        Policy Evaluation algorithm.

        Parameters
        ----------
        pi : dict
            Policy mapping states to actions.
        prev_V : np.ndarray
            Previous state values array.
        gamma : float, optional
            Discount factor, by default 1.0.
        theta : float, optional
            Convergence criterion, by default 1e-10.

        Returns
        -------
        np.ndarray
            State values array.
        """
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V, gamma=1.0):
        """
        Policy Improvement algorithm.

        Parameters
        ----------
        V : np.ndarray
            State values array.
        gamma : float, optional
            Discount factor, by default 1.0.

        Returns
        -------
        dict
            Policy mapping states to actions.
        """
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        new_pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return new_pi
