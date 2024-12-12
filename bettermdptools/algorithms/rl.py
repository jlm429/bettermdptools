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
Class that contains functions related to reinforcement learning algorithms. RL init expects an OpenAI environment (env).

Model-free learning algorithms: Q-Learning and SARSA
work out of the box with any gymnasium environments that 
have single discrete valued state spaces, like frozen lake. A lambda function 
is required to convert state spaces not in this format.
"""

import warnings

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
        self.select_action = (
            lambda state, Q, epsilon: np.random.choice(
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
            Initial value of the quantity being decayed.
        min_value : float
            Minimum value init_value is allowed to decay to.
        decay_ratio : float
            The exponential factor exp(decay_ratio).
        max_steps : int
            Max iteration steps for decaying init_value.
        log_start : float, optional
            Starting value of the decay sequence, by default -2.
        log_base : float, optional
            Base of the log space, by default 10.

        Returns
        -------
        np.ndarray
            Decay values where values[i] is the value used at i-th step.
        """
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[
            ::-1
        ]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
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
            Decay schedule of learning rate for future iterations, by default 0.5.
        init_epsilon : float, optional
            Initial epsilon value for epsilon greedy strategy, by default 1.0.
        min_epsilon : float, optional
            Minimum epsilon, by default 0.1.
        epsilon_decay_ratio : float, optional
            Decay schedule of epsilon for future iterations, by default 0.9.
        n_episodes : int, optional
            Number of episodes for the agent, by default 10000.

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
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state)
            total_reward = 0
            while not done:
                if self.render:
                    warnings.warn(
                        "Occasional render has been deprecated by openAI.  Use test_env.py to render."
                    )
                action = self.select_action(state, Q, epsilons[e])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if truncated:
                    warnings.warn(
                        "Episode was truncated.  TD target value may be incorrect."
                    )
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                td_target = reward + gamma * Q[next_state].max() * (not done)
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
            Decay schedule of learning rate for future iterations, by default 0.5.
        init_epsilon : float, optional
            Initial epsilon value for epsilon greedy strategy, by default 1.0.
        min_epsilon : float, optional
            Minimum epsilon, by default 0.1.
        epsilon_decay_ratio : float, optional
            Decay schedule of epsilon for future iterations, by default 0.9.
        n_episodes : int, optional
            Number of episodes for the agent, by default 10000.

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
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state)
            action = self.select_action(state, Q, epsilons[e])
            total_reward = 0
            while not done:
                if self.render:
                    warnings.warn(
                        "Occasional render has been deprecated by openAI.  Use test_env.py to render."
                    )
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if truncated:
                    warnings.warn(
                        "Episode was truncated.  TD target value may be incorrect."
                    )
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                next_action = self.select_action(next_state, Q, epsilons[e])
                td_target = reward + gamma * Q[next_state][next_action] * (not done)
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
