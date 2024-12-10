import gymnasium as gym

from bettermdptools.envs.acrobot_model import DiscretizedAcrobot


class CustomTransformObservation(gym.ObservationWrapper):
    def __init__(self, env, func, observation_space):
        """
        Helper class that modifies the observation space. The v26 gymnasium TransformObservation wrapper does not
        accept an observation_space parameter, which is needed in order to match the lambda conversion (tuple->int).
        Instead, we subclass gym.ObservationWrapper (parent class of gym.TransformObservation)
        to set both the conversion function and new observation space.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment to be wrapped
        func : lambda
            Function that converts the observation
        observation_space : gymnasium.spaces.Space
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
        ----------
        observation : Tuple
            Base environment observation tuple

        Returns
        -------
        int
            The converted observation (int).
        """
        return self.func(observation)


class AcrobotWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        angular_resolution_rad=0.01,
        angular_vel_resolution_rad_per_sec=0.05,
        angle_bins=None,
        velocity_bins=None,
        precomputed_P=None,
    ):
        """
        Cartpole wrapper that modifies the observation space and creates a transition/reward matrix P.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment
        angular_resolution_rad : float
            The resolution of angle bins in radians.
        angular_vel_resolution_rad_per_sec : float
            The resolution of angular velocity bins in radians per second.
        angle_bins : int, optional
            Number of discrete bins for the angles.
        velocity_bins : int, optional
            Number of discrete bins for the velocities.
        precomputed_P : dict, optional
            Precomputed transition probability matrix.
        """
        acro = DiscretizedAcrobot(
            angular_resolution_rad=angular_resolution_rad,
            angular_vel_resolution_rad_per_sec=angular_vel_resolution_rad_per_sec,
            angle_bins=angle_bins,
            velocity_bins=velocity_bins,
            precomputed_P=precomputed_P,
        )
        self._P = acro.P
        self._transform_obs = acro.transform_obs
        env = CustomTransformObservation(
            env, self._transform_obs, gym.spaces.Discrete(acro.n_states)
        )
        super().__init__(env)

    @property
    def P(self):
        """
        Returns
        -------
        dict
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns
        -------
        lambda
        """
        return self._transform_obs
