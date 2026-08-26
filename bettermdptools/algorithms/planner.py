"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
modified by: John Mansfield

documentation added by: Gagandeep Randhawa

Planning algorithms, including Value Iteration and Policy Iteration.
Planner expects a Gymnasium-style nested transition and reward dictionary P,
where P[state][action] is a list of tuples (probability, next state, reward, terminal).
When undiscounted action values tie, policy extraction uses terminal progress as
a secondary criterion so a zero-reward loop does not mask a terminating optimum.

Model-based learning algorithms: Value Iteration and Policy Iteration
"""

import warnings
from numbers import Integral

import numpy as np


def _validate_iteration_count(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


class Planner:
    def __init__(self, P):
        """
        Initializes the Planner class.

        Parameters
        ----------
        P : dict
            Transition probability matrix where P[state][action] is a list of tuples
            (probability, next state, reward, terminal).
        """
        self.P = P

    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10, dtype=np.float32):
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
            V : np.ndarray
                State values array.
            V_track : np.ndarray
                Log of V(s) for each iteration.
            pi : dict
                Policy mapping states to actions.
        """
        n_iters = _validate_iteration_count("n_iters", n_iters, 2)

        V = np.zeros(len(self.P), dtype=dtype)
        V_track = np.zeros((n_iters, len(self.P)), dtype=dtype)
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=dtype)
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

        if gamma == 1.0:
            pi = self._extract_undiscounted_policy(Q, dtype)
        else:
            pi = dict(enumerate(np.argmax(Q, axis=1)))
        return V, V_track, pi

    def value_iteration_vectorized(
        self, gamma=1.0, n_iters=1000, theta=1e-10, dtype=np.float32
    ):
        """
        Vectorized Value Iteration algorithm.

        Parameters
        ----------
        gamma : float
            Discount factor

        n_iters : int
            Number of iterations

        theta : float
            Convergence criterion for value iteration.
            State values converge when the maximum difference between new and
            previous values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.

        Returns
        -------
        tuple
            V : np.ndarray
                State values array.
            V_track : np.ndarray
                Log of V(s) for each iteration.
            pi : dict
                Policy mapping states to actions.
        """
        n_iters = _validate_iteration_count("n_iters", n_iters, 2)

        S = len(self.P)
        A = len(self.P[0])

        max_K = max(len(self.P[s][a]) for s in range(S) for a in range(A))

        prob_array = np.zeros((S, A, max_K), dtype=dtype)
        next_state_array = np.zeros((S, A, max_K), dtype=np.int32)
        reward_array = np.zeros((S, A, max_K), dtype=dtype)
        done_array = np.zeros((S, A, max_K), dtype=bool)
        mask_array = np.zeros((S, A, max_K), dtype=bool)

        for s in range(S):
            for a in range(A):
                transitions = self.P[s][a]
                for k, (prob, next_state, reward, done) in enumerate(transitions):
                    prob_array[s, a, k] = prob
                    next_state_array[s, a, k] = next_state
                    reward_array[s, a, k] = reward
                    done_array[s, a, k] = done
                    mask_array[s, a, k] = True

        V = np.zeros(S, dtype=dtype)
        V_track = np.zeros((n_iters, S), dtype=dtype)
        converged = False
        # Simpler way to handle done states
        not_done_array = 1 - done_array
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            Q = np.sum(
                prob_array
                * (reward_array + gamma * V[next_state_array] * not_done_array)
                * mask_array,
                axis=2,
            )
            V_new = np.max(Q, axis=1)

            if np.max(np.abs(V - V_new)) < theta:
                converged = True

            V = V_new
            V_track[i] = V

        if not converged:
            warnings.warn("Max iterations reached before convergence. Check n_iters.")

        if gamma == 1.0:
            pi = self._extract_undiscounted_policy(Q, dtype)
        else:
            pi = dict(enumerate(np.argmax(Q, axis=1)))
        return V, V_track, pi

    def policy_iteration(
        self,
        gamma=1.0,
        n_iters=50,
        theta=1e-10,
        dtype=np.float32,
        eval_n_iters=1000,
    ):
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
        eval_n_iters : int, optional
            Maximum Bellman sweeps per policy evaluation, by default 1000.
            Bounding evaluation prevents an improper undiscounted policy from
            blocking policy improvement indefinitely.

        Returns
        -------
        tuple
            V : np.ndarray
                State values array.
            V_track : np.ndarray
                Log of V(s) for each iteration.
            pi : dict
                Policy mapping states to actions.
        """
        n_iters = _validate_iteration_count("n_iters", n_iters, 2)
        eval_n_iters = _validate_iteration_count("eval_n_iters", eval_n_iters, 1)

        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=dtype)
        V_track = np.zeros((n_iters, len(self.P)), dtype=dtype)
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            old_pi = pi
            V, evaluation_converged = self._evaluate_policy(
                pi,
                V,
                gamma=gamma,
                theta=theta,
                dtype=dtype,
                n_iters=eval_n_iters,
            )
            V_track[i] = V
            pi = self.policy_improvement(V, gamma=gamma, dtype=dtype)
            if old_pi == pi:
                converged = True
                if not evaluation_converged:
                    warnings.warn(
                        "Policy stabilized before policy evaluation converged."
                    )

        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        return V, V_track, pi

    def policy_evaluation(
        self,
        pi,
        prev_V,
        gamma=1.0,
        theta=1e-10,
        dtype=np.float32,
        n_iters=1000,
    ):
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
        n_iters : int, optional
            Maximum Bellman sweeps, by default 1000.

        Returns
        -------
        np.ndarray
            State values array.
        """
        n_iters = _validate_iteration_count("n_iters", n_iters, 1)

        V, converged = self._evaluate_policy(
            pi,
            prev_V,
            gamma=gamma,
            theta=theta,
            dtype=dtype,
            n_iters=n_iters,
        )
        if not converged:
            warnings.warn("Max iterations reached before policy evaluation converged.")
        return V

    def _evaluate_policy(self, pi, prev_V, gamma, theta, dtype, n_iters):
        for _ in range(n_iters):
            V = np.zeros(len(self.P), dtype=dtype)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                return V, True
            prev_V = V.copy()
        return V, False

    def policy_improvement(self, V, gamma=1.0, dtype=np.float32):
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
        Q = self._action_values(V, gamma=gamma, dtype=dtype)

        if gamma == 1.0:
            return self._extract_undiscounted_policy(Q, dtype)
        return dict(enumerate(np.argmax(Q, axis=1)))

    def _action_values(self, V, gamma, dtype):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=dtype)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        return Q

    def _extract_undiscounted_policy(self, Q, dtype):
        """Prefer terminal progress when primary action values are tied."""
        maxima = np.max(Q, axis=1)
        eligible = Q == maxima[:, None]

        secondary_gamma = 0.99
        secondary_values = np.full(
            len(self.P),
            -1.0 / (1.0 - secondary_gamma),
            dtype=np.float64,
        )
        secondary_Q = np.full(Q.shape, -np.inf, dtype=np.float64)

        while True:
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    if not eligible[s, a]:
                        secondary_Q[s, a] = -np.inf
                        continue
                    secondary_Q[s, a] = sum(
                        prob
                        * (
                            -1.0
                            + secondary_gamma
                            * secondary_values[next_state]
                            * (not done)
                        )
                        for prob, next_state, _, done in self.P[s][a]
                    )

            next_secondary_values = np.max(secondary_Q, axis=1)
            if np.max(np.abs(secondary_values - next_secondary_values)) < 1e-10:
                break
            secondary_values = next_secondary_values

        return dict(enumerate(np.argmax(secondary_Q, axis=1)))
