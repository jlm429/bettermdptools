import gymnasium as gym

from bettermdptools.envs.acrobot_model import DiscretizedAcrobot


class CustomTransformObservation(gym.wrappers.TransformObservation):
    """Transform observations while declaring the transformed space."""

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


class AcrobotWrapper(gym.Wrapper):
    """Wrap Gymnasium Acrobot with discrete observations and a tabular model."""

    def __init__(
        self,
        env,
        angular_resolution_rad=None,
        angular_vel_resolution_rad_per_sec=None,
        angle_bins=None,
        velocity_bins=None,
        precomputed_P=None,
    ):
        """
        Modify the observation space and create transition and reward matrix P.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment
        angular_resolution_rad : float, optional
            The resolution of angle bins in radians. If neither this nor
            ``angle_bins`` is given, ten bins are used.
        angular_vel_resolution_rad_per_sec : float, optional
            The resolution of angular velocity bins in radians per second. If
            neither this nor ``velocity_bins`` is given, ten bins are used for
            each velocity dimension.
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
