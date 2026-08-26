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


def _identity(value):
    return value


class TestEnv:
    """Simulate rollouts using a learned or user-driven policy."""

    __test__ = False

    @staticmethod
    def test_env(
        env,
        desc=None,
        render=False,
        n_iters=10,
        pi=None,
        user_input=False,
        convert_state_obs=_identity,
        seed=None,
    ):
        """
        Simulate episodes using a policy and return the total reward from each episode.

        Parameters
        ----------
        env : gymnasium.Env
            Gymnasium environment instance.
        desc : np.ndarray, optional
            Description used by environments such as custom FrozenLake maps.
            Only used when `render=True` causes the environment to be re-created.
        render : bool, default False
            If True, use the supplied environment when it was created with
            `render_mode="human"`. Otherwise, re-create an equivalent
            environment in human render mode when its wrapper specification is
            reproducible.
        n_iters : int, default 10
            Number of episodes to simulate.
        pi : array-like or callable, optional
            Policy mapping states to actions, commonly indexed as `pi[state]`.
            If `user_input=True`, this is shown as a suggested action.
        user_input : bool, default False
            If True, prompt the user to select each action interactively.
        convert_state_obs : callable or None, default identity
            Converts observations into discrete or transformed states.
            If None, the observation is used directly.
        seed : int, optional
            Seed passed to the first environment reset. Later resets continue
            the environment's seeded random number sequence.

        Returns
        -------
        np.ndarray
            Array of length `n_iters` containing the total reward for each episode.

        Notes
        -----
        - This function assumes a discrete action space with `env.action_space.n`.
        - Internally created rendered environments are closed. A supplied
          human-rendering environment remains owned by the caller.
        - A wrapped environment that cannot be re-created must be constructed
          with `render_mode="human"` before its wrappers are applied.
        """
        if convert_state_obs is None:
            convert_state_obs = _identity

        created_env = False
        if render and getattr(env, "render_mode", None) != "human":
            env_spec = env.spec
            if env_spec is None:
                raise ValueError(
                    "render=True requires an environment with a Gymnasium spec or "
                    "an environment created with render_mode='human'"
                )
            make_kwargs = {"render_mode": "human"}
            if desc is not None:
                make_kwargs["desc"] = desc
            if any(
                wrapper_spec.kwargs is None
                for wrapper_spec in env_spec.additional_wrappers
            ):
                raise ValueError(
                    "render=True cannot re-create this wrapper stack. Create "
                    "the base environment with render_mode='human', apply the "
                    "same wrappers, and pass that wrapped environment."
                )
            try:
                env = gym.make(env_spec, **make_kwargs)
            except ValueError as exc:
                if env_spec.additional_wrappers:
                    raise ValueError(
                        "render=True cannot re-create this wrapper stack. Create "
                        "the base environment with render_mode='human', apply the "
                        "same wrappers, and pass that wrapped environment."
                    ) from exc
                raise
            created_env = True

        try:
            n_actions = env.action_space.n
            test_scores = np.full(n_iters, np.nan, dtype=float)

            for i in range(n_iters):
                if i == 0 and seed is not None:
                    state, info = env.reset(seed=seed)
                else:
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

            return test_scores
        finally:
            if created_env:
                env.close()

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
