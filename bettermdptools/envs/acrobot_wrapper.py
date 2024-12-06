"""
Author: Aleksandr Spiridonov
BSD 3-Clause License
"""

import gymnasium as gym
from bettermdptools.envs.acrobot_discretized import DiscretizedAcrobot

class CustomTransformObservation(gym.ObservationWrapper):
    def __init__(self, env, func, observation_space):
        super().__init__(env)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def observation(self, observation):
        return self.func(observation)
    
class AcrobotWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 angle_bins=10,
                angular_velocity_bins=10,
    ):
        super().__init__(env)

        self.discretized_acrobat = DiscretizedAcrobot(
            angle_bins=angle_bins,
            angular_velocity_bins=angular_velocity_bins
        )

        self._P = self.discretized_acrobat.P

        self._transform_obs = self.discretized_acrobat.transform_cont_obs

        self.observation_space = gym.spaces.Discrete(self.discretized_acrobat.state_space)
        self.env = CustomTransformObservation(env, self._transform_obs, self.observation_space)
        self.gym_env = env

        self.action_space = gym.spaces.Discrete(self.discretized_acrobat.action_space)

    @property
    def P(self):
        return self._P
    
    @property
    def transform_obs(self):
        return self._transform_obs
    
    @property
    def get_action_value(self):
        f = lambda action: action
        return f
    
def get_env_str(angle_bins, angular_velocity_bins):
    return f'acrobot_{angle_bins}_{angular_velocity_bins}'
    
def init_wrapper_env(angle_bins=10, angular_velocity_bins=10):

    acrobot_genv_train = gym.make('Acrobot-v1')

    acrobot_train = AcrobotWrapper(
        angle_bins=angle_bins,
        angular_velocity_bins=angular_velocity_bins,
        env=acrobot_genv_train
    )

    return acrobot_train