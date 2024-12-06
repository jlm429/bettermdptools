"""
Author: Aleksandr Spiridonov
BSD 3-Clause License
"""

import gymnasium as gym
from bettermdptools.envs.pendulum_discretized import DiscretizedPendulum  # Ensure this path is correct


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


class PendulumWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 angle_bins=11,
                 angular_velocity_bins=11,
                 torque_bins=11):
        """
        Pendulum wrapper that modifies the observation and action spaces and creates a transition/reward matrix P.

        Parameters
        ----------------------------
        env {gymnasium.Env}: Base environment
        angle_bins (int): Number of discrete bins for the pendulum's angle.
        angular_velocity_bins (int): Number of discrete bins for the pendulum's angular velocity.
        action_bins (int): Number of discrete bins for the torque action.
        angular_center_resolution (float): The resolution of angle bins near the center (around zero).
        angular_outer_resolution (float): The resolution of angle bins away from the center.
        torque_range (tuple): The minimum and maximum torque values.
        """
        super().__init__(env)

        # Initialize the DiscretizedPendulum model
        self.discretized_pendulum = DiscretizedPendulum(
            angle_bins=angle_bins,
            angular_velocity_bins=angular_velocity_bins,
            torque_bins=torque_bins,
        )

        # Transition probability matrix
        self._P = self.discretized_pendulum.P

        # Transformation function from continuous to discrete observations
        self._transform_obs = self.discretized_pendulum.transform_cont_obs
        self._get_action_value = self.discretized_pendulum.get_action_value

        # Wrap the environment's observation space
        self.observation_space = gym.spaces.Discrete(self.discretized_pendulum.state_space)
        self.env = CustomTransformObservation(env, self._transform_obs, self.observation_space)
        self.gym_env = env

        # Override the action space to be discrete
        self.action_space = gym.spaces.Discrete(self.discretized_pendulum.action_space)

    @property
    def P(self):
        """
        Returns the transition probability matrix.

        Returns
        ----------------------------
        _P {dict}
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns the observation transformation function.

        Returns
        ----------------------------
        _transform_obs {lambda}
        """
        return self._transform_obs
    
    @property
    def get_action_value(self):
        f = lambda action: [self._get_action_value(action)]
        return f

    def step(self, action):
        """
        Takes a discrete action, maps it to a continuous torque, and interacts with the environment.

        Parameters
        ----------------------------
        action {int}:
            The discrete action index.

        Returns
        ----------------------------
        state {int}:
            The discretized next state index.
        reward {float}:
            The reward obtained from the environment.
        done {bool}:
            Whether the episode has terminated.
        info {dict}:
            Additional information from the environment.
        """
        # Map discrete action to continuous torque
        torque = self.discretized_pendulum.get_action_value(action)

        return self.env.step([torque])

def get_env_str(angle_bins, angular_velocity_bins, torque_bins):
    """
    Returns the environment string based on the discretization parameters.

    Parameters
    ----------------------------
    angle_bins (int): Number of discrete bins for the pendulum's angle.
    angular_velocity_bins (int): Number of discrete bins for the pendulum's angular velocity.
    action_bins (int): Number of discrete bins for the torque action.

    Returns
    ----------------------------
    env_str {str}: The environment string.
    """
    return f'pendulum_{angle_bins}_{angular_velocity_bins}_{torque_bins}'

def init_wrapper_env(angle_bins=11, angular_velocity_bins=11, torque_bins=11):
    """
    Initializes the Pendulum wrapper environment.

    Parameters
    ----------------------------
    angle_bins (int): Number of discrete bins for the pendulum's angle.
    angular_velocity_bins (int): Number of discrete bins for the pendulum's angular velocity.
    torque_bins (int): Number of discrete bins for the torque action.

    Returns
    ----------------------------
    pendulum_env {PendulumWrapper}: The Pendulum wrapper environment.
    """
    pendulum_genv_train = gym.make('Pendulum-v1')

    pendulum_train = PendulumWrapper(
        angle_bins=angle_bins,
        angular_velocity_bins=angular_velocity_bins,
        torque_bins=torque_bins,
        env=pendulum_genv_train
    )

    return pendulum_train