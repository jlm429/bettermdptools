"""Discretized CartPole transition model with adaptive angle binning."""

import numpy as np


class DiscretizedCartPole:
    """Build a deterministic tabular approximation of CartPole dynamics."""

    def __init__(
        self,
        position_bins,
        velocity_bins,
        angular_velocity_bins,
        threshold_bins,
        angular_center_resolution,
        angular_outer_resolution,
        sutton_barto_reward=False,
    ):
        """
        Initializes the DiscretizedCartPole model.

        Parameters:
        - position_bins (int): Number of discrete bins for the cart's position.
        - velocity_bins (int): Number of discrete bins for the cart's velocity.
        - angular_velocity_bins (int): Number of angular velocity bins.
        - threshold_bins (float): Step size for binned physics calculations.
        - angular_center_resolution (float): Angle resolution near zero.
        - angular_outer_resolution (float): Angle resolution away from zero.
        - sutton_barto_reward (bool): Whether to use Sutton and Barto rewards.

        Attributes:
        - state_space (int): Total number of discrete states in the environment.
        - P (dict): Transition matrix whose entries contain probability, next
          state, reward, and termination status.
        - transform_obs (lambda): Converts observations to state indices.
        """
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.threshold_bins = threshold_bins
        self.action_space = 2  # Left or Right
        self.sutton_barto_reward = sutton_barto_reward

        # Define the range for each variable
        self.position_range = (-2.4, 2.4)
        self.velocity_range = (-3, 3)
        self.angle_range = (-12 * np.pi / 180, 12 * np.pi / 180)
        self.angular_velocity_range = (-1.5, 1.5)
        self.angular_center_resolution = angular_center_resolution
        self.angular_outer_resolution = angular_outer_resolution

        # Use adaptive binning for the pole angle
        self.angle_bins = self.adaptive_angle_bins(
            self.angle_range,
            self.angular_center_resolution,
            self.angular_outer_resolution,
        )  # Adjust these values as needed

        self.state_space = np.prod(
            [
                self.position_bins,
                self.velocity_bins,
                len(self.angle_bins),
                self.angular_velocity_bins,
            ]
        )
        self.P = {
            state: {action: [] for action in range(self.action_space)}
            for state in range(self.state_space)
        }
        self.setup_transition_probabilities()
        self.n_states = (
            len(self.angle_bins)
            * self.velocity_bins
            * self.position_bins
            * self.angular_velocity_bins
        )
        """
        Explanation of transform_obs lambda:
        This function determines which bins contain CartPole observations,
        and then convert bin coordinates into a single index.  This makes it possible
        to use tabular learning and planning with continuous state spaces.
        
        Parameters:
        - obs (list): Position, velocity, angle, and angular velocity.

        Returns:
        - int: A single integer representing the discretized state index.
        """
        self.transform_obs = lambda obs: (
            np.ravel_multi_index(
                (
                    np.clip(
                        np.digitize(
                            obs[0],
                            np.linspace(*self.position_range, self.position_bins),
                        )
                        - 1,
                        0,
                        self.position_bins - 1,
                    ),
                    np.clip(
                        np.digitize(
                            obs[1],
                            np.linspace(*self.velocity_range, self.velocity_bins),
                        )
                        - 1,
                        0,
                        self.velocity_bins - 1,
                    ),
                    np.clip(
                        np.digitize(obs[2], self.angle_bins) - 1,
                        0,
                        len(self.angle_bins) - 1,
                    ),
                    # Use adaptive angle bins
                    np.clip(
                        np.digitize(
                            obs[3],
                            np.linspace(
                                *self.angular_velocity_range, self.angular_velocity_bins
                            ),
                        )
                        - 1,
                        0,
                        self.angular_velocity_bins - 1,
                    ),
                ),
                (
                    self.position_bins,
                    self.velocity_bins,
                    len(self.angle_bins),
                    self.angular_velocity_bins,
                ),
            )
        )

    def adaptive_angle_bins(self, angle_range, center_resolution, outer_resolution):
        """
        Generate adaptive angle bins with finer resolution near the center.

        Parameters:
        - angle_range (tuple): The minimum and maximum angles in radians.
        - center_resolution (float): Bin width near zero angle for higher resolution.
        - outer_resolution (float): Bin width away from zero for lower resolution.

        Returns:
        - np.array: An array of bin edges with adaptive spacing.
        """
        min_angle, max_angle = angle_range
        # Generate finer bins around zero
        center_bins = np.arange(
            -center_resolution, center_resolution + 1e-6, center_resolution / 10
        )
        # Generate sparser bins outside the center region
        left_bins = np.linspace(
            min_angle,
            -center_resolution,
            num=int((-center_resolution - min_angle) / outer_resolution) + 1,
            endpoint=False,
        )
        right_bins = np.linspace(
            center_resolution,
            max_angle,
            num=np.max(
                [int((max_angle - center_resolution) / outer_resolution) + 1, 2]
            ),
            endpoint=True,
        )[1:]
        return np.unique(np.concatenate([left_bins, center_bins, right_bins]))

    def setup_transition_probabilities(self):
        """
        Iterate through all states and actions and record their deterministic
        transitions, rewards, and termination status.
        """
        for state in range(self.state_space):
            position, velocity, angle, angular_velocity = self.index_to_state(state)
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(
                    position, velocity, angle, angular_velocity, action
                )
                self.P[state][action].append((1, next_state, reward, done))

    def index_to_state(self, index):
        """
        Converts a single index into a multidimensional state representation.

        Parameters:
        - index (int): The flat index representing the state.

        Returns:
        - list: Position, velocity, angle, and angular velocity bin indices.
        """
        totals = [
            self.position_bins,
            self.velocity_bins,
            len(self.angle_bins),
            self.angular_velocity_bins,
        ]
        multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        components = [int((index // multipliers[i]) % totals[i]) for i in range(4)]
        return components

    def compute_next_state(
        self, position_idx, velocity_idx, angle_idx, angular_velocity_idx, action
    ):
        """
        Compute the next state from state indices and an action using simplified
        physics.

        Parameters:
        - position_idx (int): Current index of the cart's position.
        - velocity_idx (int): Current index of the cart's velocity.
        - angle_idx (int): Current index of the pole's angle.
        - angular_velocity_idx (int): Current index of the pole's angular velocity.
        - action (int): Action taken (0 for left, 1 for right).

        Returns:
        - tuple: Next state index, reward, and episode termination flag.
        """
        position = np.linspace(*self.position_range, self.position_bins)[position_idx]
        velocity = np.linspace(*self.velocity_range, self.velocity_bins)[velocity_idx]
        angle = self.angle_bins[angle_idx]
        angular_velocity = np.linspace(
            *self.angular_velocity_range, self.angular_velocity_bins
        )[angular_velocity_idx]

        # Simulate physics here (simplified)
        thresh = self.threshold_bins
        force = 10 if action == 1 else -10
        new_velocity = velocity + (force + np.sin(angle) * -10.0) * thresh
        new_position = position + new_velocity * thresh
        new_angular_velocity = angular_velocity + (3.0 * np.sin(angle) - force) * thresh
        new_angle = angle + new_angular_velocity * thresh

        new_position_idx = np.clip(
            np.digitize(
                new_position, np.linspace(*self.position_range, self.position_bins)
            )
            - 1,
            0,
            self.position_bins - 1,
        )
        new_velocity_idx = np.clip(
            np.digitize(
                new_velocity, np.linspace(*self.velocity_range, self.velocity_bins)
            )
            - 1,
            0,
            self.velocity_bins - 1,
        )
        new_angle_idx = np.clip(
            np.digitize(new_angle, self.angle_bins) - 1, 0, len(self.angle_bins) - 1
        )
        new_angular_velocity_idx = np.clip(
            np.digitize(
                new_angular_velocity,
                np.linspace(*self.angular_velocity_range, self.angular_velocity_bins),
            )
            - 1,
            0,
            self.angular_velocity_bins - 1,
        )

        new_state_idx = np.ravel_multi_index(
            (
                new_position_idx,
                new_velocity_idx,
                new_angle_idx,
                new_angular_velocity_idx,
            ),
            (
                self.position_bins,
                self.velocity_bins,
                len(self.angle_bins),
                self.angular_velocity_bins,
            ),
        )

        done = bool(np.abs(new_angle) > 12 * np.pi / 180 or np.abs(new_position) > 2.4)
        reward = -1.0 if self.sutton_barto_reward and done else 0.0
        if not self.sutton_barto_reward:
            reward = 1.0

        return new_state_idx, reward, done
