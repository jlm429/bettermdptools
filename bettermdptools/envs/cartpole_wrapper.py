"""
Author: John Mansfield
BSD 3-Clause License
"""

import gymnasium as gym
import numpy as np
from bettermdptools.envs.cartpole_model import DiscretizedCartPole


class CustomTransformObservation(gym.ObservationWrapper):
    def __init__(self, env, func, observation_space):
        """
        Helper class that modifies the observation space. The v26 gymnasium TransformObservation wrapper does not
        accept an observation_space parameter, which is needed in order to match the lambda conversion (tuple->int).
        Instead, we subclass gym.ObservationWrapper (parent class of gym.TransformObservation)
        to set both the conversion function and new observation space.

        Parameters
        ----------------------------
        env {gymnasium.Env}:
            Base environment to be wrapped

        func {lambda}:
            Function that converts the observation

        observation_space {gymnasium.spaces.Space}:
            New observation space
        """
        super().__init__(env)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def observation(self, observation):
        """
        Applies a function to the observation received from the environment's step function,
        which is passed back to the user.

        Parameters
        ----------------------------
        observation {Tuple}:
            Base environment observation tuple

        Returns
        ----------------------------
        func(observation) {int}:
            The converted observation (int).
        """
        return self.func(observation)


class CartpoleWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 position_bins=10,
                 velocity_bins=10,
                 angular_velocity_bins=10,
                 angular_center_resolution=.1,
                 angular_outer_resolution=.5):
        """
        Cartpole wrapper that modifies the observation space and creates a transition/reward matrix P.

        Parameters
        ----------------------------
        env {gymnasium.Env}:
            Base environment
        """
        dpole = DiscretizedCartPole(position_bins=position_bins,
                                    velocity_bins=velocity_bins,
                                    angular_velocity_bins=angular_velocity_bins,
                                    angular_center_resolution=angular_center_resolution,
                                    angular_outer_resolution=angular_outer_resolution)
        self._P = dpole.P
        self._transform_obs = dpole.transform_obs
        env = CustomTransformObservation(env, self._transform_obs, gym.spaces.Discrete(dpole.n_states))
        super().__init__(env)

    @property
    def P(self):
        """
        Returns
        ----------------------------
        _P {dict}
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns
        ----------------------------
        _transform_obs {lambda}
        """
        return self._transform_obs
