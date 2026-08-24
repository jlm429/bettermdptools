"""
Author: Aleksandr Spiridonov
BSD 3-Clause License
"""

import gymnasium as gym
import numpy as np

from bettermdptools.envs.pendulum_discretized import (
    DiscretizedPendulum,
)  # Ensure this path is correct


class CustomTransformObservation(gym.wrappers.TransformObservation):
    def __init__(self, env, func, observation_space):
        """
        Transform observations while declaring the transformed observation space.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment to be wrapped
        func : lambda
            Function that converts the observation
        observation_space : gymnasium.spaces.Space
            New observation space
        """
        super().__init__(env=env, func=func, observation_space=observation_space)


class PendulumWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        angle_bins=11,
        angular_velocity_bins=11,
        torque_bins=11,
        n_workers=4,
        cache_dir="./cached",
        dim_samples=11,
    ):
        """
        Pendulum wrapper that modifies the observation and action spaces and creates a transition/reward matrix P.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment
        angle_bins : int
            Number of discrete bins for the pendulum's angle.
        angular_velocity_bins : int
            Number of discrete bins for the pendulum's angular velocity.
        torque_bins : int
            Number of discrete bins for the torque action.
        n_workers : int
            Number of workers used to generate the transition model.
        cache_dir : str
            Directory used to cache the generated transition model.
        dim_samples : int
            Samples per modeled state dimension.
        """
        # Initialize the DiscretizedPendulum model
        self.discretized_pendulum = DiscretizedPendulum(
            angle_bins=angle_bins,
            angular_velocity_bins=angular_velocity_bins,
            torque_bins=torque_bins,
            n_workers=n_workers,
            cache_dir=cache_dir,
            dim_samples=dim_samples,
        )

        # Transition probability matrix
        self._P = self.discretized_pendulum.P

        # Transformation function from continuous to discrete observations
        self._transform_obs = self.discretized_pendulum.transform_cont_obs
        self._get_action_value = self.discretized_pendulum.get_action_value

        # Wrap the environment's observation space
        observation_space = gym.spaces.Discrete(self.discretized_pendulum.state_space)
        transformed_env = CustomTransformObservation(
            env, self._transform_obs, observation_space
        )
        super().__init__(transformed_env)
        self.gym_env = env

        # Override the action space to be discrete
        self.action_space = gym.spaces.Discrete(self.discretized_pendulum.action_space)

    @property
    def P(self):
        """
        Returns the transition probability matrix.

        Returns
        -------
        dict
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns the observation transformation function.

        Returns
        -------
        lambda
        """
        return self._transform_obs

    @property
    def get_action_value(self):
        def action_value(action):
            return [self._get_action_value(action)]

        return action_value

    def step(self, action):
        """
        Takes a discrete action, maps it to a continuous torque, and interacts with the environment.

        Parameters
        ----------
        action : int
            The discrete action index.

        Returns
        -------
        tuple
            Discretized observation, reward, terminated, truncated, and info.
        """
        # Map discrete action to continuous torque
        torque = self.discretized_pendulum.get_action_value(action)

        continuous_action = np.asarray([torque], dtype=self.env.action_space.dtype)
        return self.env.step(continuous_action)


def get_env_str(angle_bins, angular_velocity_bins, torque_bins):
    """
    Returns the environment string based on the discretization parameters.

    Parameters
    ----------
    angle_bins : int
        Number of discrete bins for the pendulum's angle.
    angular_velocity_bins : int
        Number of discrete bins for the pendulum's angular velocity.
    torque_bins : int
        Number of discrete bins for the torque action.

    Returns
    -------
    str
        The environment string.
    """
    return f"pendulum_{angle_bins}_{angular_velocity_bins}_{torque_bins}"


def init_wrapper_env(angle_bins=11, angular_velocity_bins=11, torque_bins=11):
    """
    Initializes the Pendulum wrapper environment.

    Parameters
    ----------
    angle_bins : int
        Number of discrete bins for the pendulum's angle.
    angular_velocity_bins : int
        Number of discrete bins for the pendulum's angular velocity.
    torque_bins : int
        Number of discrete bins for the torque action.

    Returns
    -------
    PendulumWrapper
        The Pendulum wrapper environment.
    """
    pendulum_genv_train = gym.make("Pendulum-v1")

    pendulum_train = PendulumWrapper(
        angle_bins=angle_bins,
        angular_velocity_bins=angular_velocity_bins,
        torque_bins=torque_bins,
        env=pendulum_genv_train,
    )

    return pendulum_train
