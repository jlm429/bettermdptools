# -*- coding: utf-8 -*-
"""
Simulation of the agent's decision process after it has learned a policy.

Author: John Mansfield
Documentation added by: Gagandeep Randhawa
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class TestEnv:
    """Utilities for simulating environment rollouts using a learned or user-driven policy."""

    @staticmethod
    def test_env(
        env,
        desc=None,
        render=False,
        n_iters=10,
        pi=None,
        user_input=False,
        convert_state_obs=lambda state: state,
    ):
        """
        Simulate episodes using a policy and return the total reward from each episode.

        Parameters
        ----------
        env : gymnasium.Env
            Gymnasium environment instance.
        desc : np.ndarray, optional
            Environment description used by some environments (for example, custom FrozenLake maps).
            Only used when `render=True` causes the environment to be re-created.
        render : bool, default False
            If True, the environment is re-created with `render_mode="human"` so it renders visually.
        n_iters : int, default 10
            Number of episodes to simulate.
        pi : array-like or callable, optional
            Policy mapping states to actions. Commonly an array where `pi[state]` gives the action.
            If `user_input=True`, this is shown as a suggested action.
        user_input : bool, default False
            If True, prompt the user to select each action interactively.
        convert_state_obs : callable or None, default identity
            Function applied to observations to convert them into discrete or transformed states.
            If None, the observation is used directly.

        Returns
        -------
        np.ndarray
            Array of length `n_iters` containing the total reward for each episode.

        Notes
        -----
        - This function assumes a discrete action space with `env.action_space.n`.
        - When `render=True`, the environment is created internally and closed before returning.
          When `render=False`, the caller is responsible for managing the environment lifecycle.
        """
        if convert_state_obs is None:
            convert_state_obs = lambda s: s

        created_env = False
        if render:
            env_name = env.unwrapped.spec.id
            make_kwargs = {"render_mode": "human"}
            if desc is not None:
                make_kwargs["desc"] = desc
            env = gym.make(env_name, **make_kwargs)
            created_env = True

        n_actions = env.action_space.n
        test_scores = np.full(n_iters, np.nan, dtype=float)

        for i in range(n_iters):
            state, info = env.reset()
            state = convert_state_obs(state)

            done = False
            total_reward = 0.0

            while not done:
                if user_input:
                    action = TestEnv._prompt_for_action(
                        state=state,
                        n_actions=n_actions,
                        pi=pi,
                    )
                else:
                    action = pi[state]

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = convert_state_obs(next_state)
                total_reward += reward

            test_scores[i] = total_reward

        if created_env:
            env.close()

        return test_scores

    @staticmethod
    def _prompt_for_action(state: Any, n_actions: int, pi: Any) -> int:
        """
        Prompt the user to select an action and return the chosen value.
        """
        print(f"state is {state}")
        if pi is not None:
            print(f"policy output is {pi[state]}")

        while True:
            raw = input(f"Please select 0 - {n_actions - 1} then hit enter:\n")
            try:
                action = int(raw)
            except ValueError:
                print("Please enter a number")
                continue

            if 0 <= action < n_actions:
                return action

            print(f"please enter a valid action, 0 - {n_actions - 1}\n")
