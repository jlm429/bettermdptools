"""
Author: John Mansfield
BSD 3-Clause License
"""

import gymnasium as gym

from bettermdptools.envs.cartpole_model import DiscretizedCartPole


class CustomTransformObservation(gym.wrappers.TransformObservation):
    def __init__(self, env, func, observation_space):
        """
        Transform observations while declaring the transformed observation space.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment to be wrapped.
        func : lambda
            Function that converts the observation.
        observation_space : gymnasium.spaces.Space
            New observation space.
        """
        super().__init__(env=env, func=func, observation_space=observation_space)


class CartpoleWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        position_bins=10,
        velocity_bins=10,
        angular_velocity_bins=10,
        threshold_bins=0.5,
        angular_center_resolution=0.1,
        angular_outer_resolution=0.5,
    ):
        """
        Modify the observation space and create transition and reward matrix P.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment.
        position_bins : int, optional
            Number of discrete bins for the cart's position.
        velocity_bins : int, optional
            Number of discrete bins for the cart's velocity.
        angular_velocity_bins : int, optional
            Number of discrete bins for the pole's angular velocity.
        threshold_bins : float, optional
            Step size for binned physics calculations.
        angular_center_resolution : float, optional
            The resolution of angle bins near the center (around zero).
        angular_outer_resolution : float, optional
            The resolution of angle bins away from the center.
        """
        dpole = DiscretizedCartPole(
            position_bins=position_bins,
            velocity_bins=velocity_bins,
            angular_velocity_bins=angular_velocity_bins,
            threshold_bins=threshold_bins,
            angular_center_resolution=angular_center_resolution,
            angular_outer_resolution=angular_outer_resolution,
            sutton_barto_reward=getattr(env.unwrapped, "_sutton_barto_reward", False),
        )
        self._P = dpole.P
        self._transform_obs = dpole.transform_obs
        env = CustomTransformObservation(
            env, self._transform_obs, gym.spaces.Discrete(dpole.n_states)
        )
        super().__init__(env)

    @property
    def P(self):
        """
        Returns
        -------
        dict
            Transition/reward matrix.
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns
        -------
        lambda
            Function that converts the observation.
        """
        return self._transform_obs
