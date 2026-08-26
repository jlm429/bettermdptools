"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
modified by: John Mansfield

documentation added by: Gagandeep Randhawa

Reinforcement learning algorithms. RL expects a Gymnasium environment.

Model-free learning algorithms: Q-Learning and SARSA
work out of the box with any gymnasium environments that
have single discrete valued state spaces, like frozen lake. A lambda function
is required to convert state spaces not in this format.
"""

import warnings
from numbers import Integral, Real

import numpy as np
from tqdm.auto import tqdm

from bettermdptools.utils.callbacks import MyCallbacks


class RL:
    def __init__(self, env):
        self.env = env
        self.callbacks = MyCallbacks()
        self.render = False
        # Explanation of lambda:
        # def select_action(state, Q, epsilon):
        #   if np.random.random() > epsilon:
        #       max_val = np.max(Q[state])
        #       indxs_selector = np.isclose(Q[state], max_val)
        #       indxs = np.arange(len(Q[state]))[indxs_selector]
        #       return np.random.choice(indxs)
        #   else:
        #       return np.random.randint(len(Q[state]))
        self.select_action = lambda state, Q, epsilon: (
            np.random.choice(
                np.arange(len(Q[state]))[np.isclose(Q[state], np.max(Q[state]))]
            )
            if np.random.random() > epsilon
            else np.random.randint(len(Q[state]))
        )

    @staticmethod
    def decay_schedule(
        init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10
    ):
        """
        Generates a decay schedule for a given initial value.

        Parameters
        ----------
        init_value : float
            Finite initial value of the quantity being decayed.
        min_value : float
            Finite minimum value, no greater than `init_value`.
        decay_ratio : float
            Fraction of `max_steps` over which values decay, in (0, 1].
        max_steps : int
            Positive number of steps in the returned schedule.
        log_start : float, optional
            Finite negative start of the logarithmic sequence, by default -2.
        log_base : float, optional
            Finite logarithm base greater than 1, by default 10.

        Returns
        -------
        np.ndarray
            Decay values where values[i] is the value used at i-th step.
        """
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, Integral)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(decay_ratio, bool)
            or not isinstance(decay_ratio, Real)
            or not np.isfinite(decay_ratio)
            or not 0 < decay_ratio <= 1
        ):
            raise ValueError("decay_ratio must be finite and in (0, 1]")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in (init_value, min_value)
        ):
            raise ValueError("init_value and min_value must be finite")
        if init_value < min_value:
            raise ValueError("init_value must be greater than or equal to min_value")
        if (
            isinstance(log_start, bool)
            or not isinstance(log_start, Real)
            or not np.isfinite(log_start)
            or log_start >= 0
        ):
            raise ValueError("log_start must be finite and negative")
        if (
            isinstance(log_base, bool)
            or not isinstance(log_base, Real)
            or not np.isfinite(log_base)
            or log_base <= 1
        ):
            raise ValueError("log_base must be finite and greater than 1")

        max_steps = int(max_steps)
        if max_steps == 1 or init_value == min_value:
            return np.full(max_steps, init_value, dtype=float)

        decay_steps = min(max_steps, max(2, int(max_steps * decay_ratio)))
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[
            ::-1
        ]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), constant_values=min_value)
        return values

    def q_learning(
        self,
        nS=None,
        nA=None,
        convert_state_obs=lambda state: state,
        gamma=0.99,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_ratio=0.5,
        init_epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay_ratio=0.9,
        n_episodes=10000,
        seed=None,
    ):
        """
        Q-Learning algorithm.

        Parameters
        ----------
        nS : int, optional
            Number of states, by default None.
        nA : int, optional
            Number of available actions, by default None.
        convert_state_obs : function, optional
            Converts state into an integer, by default lambda state: state.
        gamma : float, optional
            Discount factor, by default 0.99.
        init_alpha : float, optional
            Initial learning rate, by default 0.5.
        min_alpha : float, optional
            Minimum learning rate, by default 0.01.
        alpha_decay_ratio : float, optional
            Learning-rate schedule ratio passed to `decay_schedule`, by default 0.5.
        init_epsilon : float, optional
            Initial epsilon value for epsilon greedy strategy, by default 1.0.
        min_epsilon : float, optional
            Minimum epsilon, by default 0.1.
        epsilon_decay_ratio : float, optional
            Exploration schedule ratio passed to `decay_schedule`, by default 0.9.
        n_episodes : int, optional
            Positive number of episodes for the agent, by default 10000.
        seed : int, optional
            Seed passed to the first environment reset. Later resets continue
            the environment's seeded random number sequence.

        Returns
        -------
        tuple
            Q : np.ndarray
                Final action-value function Q(s,a).
            V : np.ndarray
                State values array.
            pi : dict
                Policy mapping states to actions.
            Q_track : np.ndarray
                Log of Q(s,a) for each episode.
            pi_track : list
                Log of complete policy for each episode.
            rewards : np.ndarray
                Rewards obtained in each episode.

        Notes
        -----
        Episodes stop after either termination or truncation. The TD target
        bootstraps across truncation, but not across a true terminal state.
        """
        if nS is None:
            nS = self.env.observation_space.n
        if nA is None:
            nA = self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float32)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float32)
        alphas = RL.decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
        epsilons = RL.decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )
        rewards = np.zeros(n_episodes, dtype=np.float32)
        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            if e == 0 and seed is not None:
                state, info = self.env.reset(seed=seed)
            else:
                state, info = self.env.reset()
            episode_done = False
            state = convert_state_obs(state)
            total_reward = 0
            while not episode_done:
                if self.render:
                    warnings.warn(
                        "Occasional rendering is deprecated. Use test_env.py "
                        "to render."
                    )
                action = self.select_action(state, Q, epsilons[e])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                td_target = reward + gamma * Q[next_state].max() * (not terminated)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
                total_reward += reward
            rewards[e] = total_reward
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)

        pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return Q, V, pi, Q_track, pi_track, rewards

    def sarsa(
        self,
        nS=None,
        nA=None,
        convert_state_obs=lambda state: state,
        gamma=0.99,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_ratio=0.5,
        init_epsilon=1.0,
        min_epsilon=0.1,
        epsilon_decay_ratio=0.9,
        n_episodes=10000,
        seed=None,
    ):
        """
        SARSA algorithm.

        Parameters
        ----------
        nS : int, optional
            Number of states, by default None.
        nA : int, optional
            Number of available actions, by default None.
        convert_state_obs : function, optional
            Converts state into an integer, by default lambda state: state.
        gamma : float, optional
            Discount factor, by default 0.99.
        init_alpha : float, optional
            Initial learning rate, by default 0.5.
        min_alpha : float, optional
            Minimum learning rate, by default 0.01.
        alpha_decay_ratio : float, optional
            Learning-rate schedule ratio passed to `decay_schedule`, by default 0.5.
        init_epsilon : float, optional
            Initial epsilon value for epsilon greedy strategy, by default 1.0.
        min_epsilon : float, optional
            Minimum epsilon, by default 0.1.
        epsilon_decay_ratio : float, optional
            Exploration schedule ratio passed to `decay_schedule`, by default 0.9.
        n_episodes : int, optional
            Positive number of episodes for the agent, by default 10000.
        seed : int, optional
            Seed passed to the first environment reset. Later resets continue
            the environment's seeded random number sequence.

        Returns
        -------
        tuple
            Q : np.ndarray
                Final action-value function Q(s,a).
            V : np.ndarray
                State values array.
            pi : dict
                Policy mapping states to actions.
            Q_track : np.ndarray
                Log of Q(s,a) for each episode.
            pi_track : list
                Log of complete policy for each episode.
            rewards : np.ndarray
                Rewards obtained in each episode.

        Notes
        -----
        Episodes stop after either termination or truncation. The TD target
        bootstraps across truncation, but not across a true terminal state.
        """
        if nS is None:
            nS = self.env.observation_space.n
        if nA is None:
            nA = self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float32)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float32)
        rewards = np.zeros(n_episodes, dtype=np.float32)
        alphas = RL.decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
        epsilons = RL.decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )

        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            if e == 0 and seed is not None:
                state, info = self.env.reset(seed=seed)
            else:
                state, info = self.env.reset()
            episode_done = False
            state = convert_state_obs(state)
            action = self.select_action(state, Q, epsilons[e])
            total_reward = 0
            while not episode_done:
                if self.render:
                    warnings.warn(
                        "Occasional rendering is deprecated. Use test_env.py "
                        "to render."
                    )
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                next_action = self.select_action(next_state, Q, epsilons[e])
                td_target = reward + gamma * Q[next_state][next_action] * (
                    not terminated
                )
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state, action = next_state, next_action
                total_reward += reward
            rewards[e] = total_reward
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)

        pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return Q, V, pi, Q_track, pi_track, rewards
