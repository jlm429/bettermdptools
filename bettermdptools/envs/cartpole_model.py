"""
generative AI experiment - discretized cartpole transition and reward (P) matrix with adaptive angle binning
created with chatGPT

Example usage:
dpole = DiscretizedCartPole(10, 10, 10, .1, .5)  # Example bin sizes for each variable and adaptive angle binning center/outer resolution

"""

import numpy as np


class DiscretizedCartPole:
    def __init__(self,
                 position_bins,
                 velocity_bins,
                 angular_velocity_bins,
                 angular_center_resolution,
                 angular_outer_resolution):

        """
         Initializes the DiscretizedCartPole model.

         Parameters:
         - position_bins (int): Number of discrete bins for the cart's position.
         - velocity_bins (int): Number of discrete bins for the cart's velocity.
         - angular_velocity_bins (int): Number of discrete bins for the pole's angular velocity.
         - angular_center_resolution (float): The resolution of angle bins near the center (around zero).
         - angular_outer_resolution (float): The resolution of angle bins away from the center.

         Attributes:
         - state_space (int): Total number of discrete states in the environment.
         - P (dict): Transition probability matrix where P[state][action] is a list of tuples (probability, next_state,
         reward, done).
         - transform_obs (lambda): Function to transform continuous observations into a discrete state index.
         """
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.action_space = 2  # Left or Right

        # Define the range for each variable
        self.position_range = (-2.4, 2.4)
        self.velocity_range = (-3, 3)
        self.angle_range = (-12 * np.pi / 180, 12 * np.pi / 180)
        self.angular_velocity_range = (-1.5, 1.5)
        self.angular_center_resolution = angular_center_resolution
        self.angular_outer_resolution = angular_outer_resolution

        # Use adaptive binning for the pole angle
        self.angle_bins = self.adaptive_angle_bins(self.angle_range, self.angular_center_resolution, self.angular_outer_resolution)  # Adjust these values as needed

        self.state_space = np.prod([self.position_bins, self.velocity_bins, len(self.angle_bins), self.angular_velocity_bins])
        self.P = {state: {action: [] for action in range(self.action_space)} for state in range(self.state_space)}
        self.setup_transition_probabilities()
        self.n_states = len(self.angle_bins)*self.velocity_bins*self.position_bins*self.angular_velocity_bins
        """
        Explanation of transform_obs lambda: 
        This lambda function will take cartpole observations, determine which bins they fall into, 
        and then convert bin coordinates into a single index.  This makes it possible 
        to use traditional reinforcement learning and planning algorithms, designed for discrete spaces, with continuous 
        state space environments. 
        
        Parameters:
        - obs (list): A list of continuous observations [position, velocity, angle, angular_velocity].

        Returns:
        - int: A single integer representing the discretized state index.
        """
        self.transform_obs = lambda obs: (
            np.ravel_multi_index((
                np.clip(np.digitize(obs[0], np.linspace(*self.position_range, self.position_bins)) - 1, 0,
                        self.position_bins - 1),
                np.clip(np.digitize(obs[1], np.linspace(*self.velocity_range, self.velocity_bins)) - 1, 0,
                        self.velocity_bins - 1),
                np.clip(np.digitize(obs[2], self.angle_bins) - 1, 0, len(self.angle_bins) - 1),
                # Use adaptive angle bins
                np.clip(np.digitize(obs[3], np.linspace(*self.angular_velocity_range, self.angular_velocity_bins)) - 1,
                        0, self.angular_velocity_bins - 1)
            ), (self.position_bins, self.velocity_bins, len(self.angle_bins), self.angular_velocity_bins))
        )

    def adaptive_angle_bins(self, angle_range, center_resolution, outer_resolution):
        """
        Generates adaptive bins for the pole's angle to allow for finer resolution near the center and coarser
        resolution farther away.

        Parameters:
        - angle_range (tuple): The minimum and maximum angles in radians.
        - center_resolution (float): Bin width near zero angle for higher resolution.
        - outer_resolution (float): Bin width away from zero for lower resolution.

        Returns:
        - np.array: An array of bin edges with adaptive spacing.
        """
        min_angle, max_angle = angle_range
        # Generate finer bins around zero
        center_bins = np.arange(-center_resolution, center_resolution + 1e-6, center_resolution / 10)
        # Generate sparser bins outside the center region
        left_bins = np.linspace(min_angle, -center_resolution,
                                num=int((center_resolution - min_angle) / outer_resolution) + 1, endpoint=False)
        right_bins = np.linspace(center_resolution, max_angle,
                                 num=int((max_angle - center_resolution) / outer_resolution) + 1, endpoint=True)
        return np.unique(np.concatenate([left_bins, center_bins, right_bins]))

    def setup_transition_probabilities(self):
        """
        Sets up the transition probabilities for the environment. This method iterates through all possible
        states and actions, simulates the next state, and records the transition probability
        (deterministic in this setup), reward, and termination status.
        """
        for state in range(self.state_space):
            position, velocity, angle, angular_velocity = self.index_to_state(state)
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(position, velocity, angle, angular_velocity, action)
                self.P[state][action].append((1, next_state, reward, done))

    def index_to_state(self, index):
        """
        Converts a single index into a multidimensional state representation.

        Parameters:
        - index (int): The flat index representing the state.

        Returns:
        - list: A list of indices representing the state in terms of position, velocity, angle, and angular velocity bins.
        """
        totals = [self.position_bins, self.velocity_bins, len(self.angle_bins), self.angular_velocity_bins]
        multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        components = [int((index // multipliers[i]) % totals[i]) for i in range(4)]
        return components

    def compute_next_state(self, position_idx, velocity_idx, angle_idx, angular_velocity_idx, action):
        """
        Computes the next state based on the current state indices and the action taken. Applies simplified physics calculations to determine the next state.

        Parameters:
        - position_idx (int): Current index of the cart's position.
        - velocity_idx (int): Current index of the cart's velocity.
        - angle_idx (int): Current index of the pole's angle.
        - angular_velocity_idx (int): Current index of the pole's angular velocity.
        - action (int): Action taken (0 for left, 1 for right).

        Returns:
        - tuple: Contains the next state index, the reward, and the done flag indicating if the episode has ended.
        """
        position = np.linspace(*self.position_range, self.position_bins)[position_idx]
        velocity = np.linspace(*self.velocity_range, self.velocity_bins)[velocity_idx]
        angle = self.angle_bins[angle_idx]
        angular_velocity = np.linspace(*self.angular_velocity_range, self.angular_velocity_bins)[angular_velocity_idx]

        # Simulate physics here (simplified)
        force = 10 if action == 1 else -10
        new_velocity = velocity + (force + np.cos(angle) * -10.0) * 0.02
        new_position = position + new_velocity * 0.02
        new_angular_velocity = angular_velocity + (-3.0 * np.sin(angle)) * 0.02
        new_angle = angle + new_angular_velocity * 0.02

        new_position_idx = np.clip(np.digitize(new_position, np.linspace(*self.position_range, self.position_bins)) - 1, 0, self.position_bins-1)
        new_velocity_idx = np.clip(np.digitize(new_velocity, np.linspace(*self.velocity_range, self.velocity_bins)) - 1, 0, self.velocity_bins-1)
        new_angle_idx = np.clip(np.digitize(new_angle, self.angle_bins) - 1, 0, len(self.angle_bins)-1)
        new_angular_velocity_idx = np.clip(np.digitize(new_angular_velocity, np.linspace(*self.angular_velocity_range, self.angular_velocity_bins)) - 1, 0, self.angular_velocity_bins-1)

        new_state_idx = np.ravel_multi_index((new_position_idx, new_velocity_idx, new_angle_idx, new_angular_velocity_idx),
                                             (self.position_bins, self.velocity_bins, len(self.angle_bins), self.angular_velocity_bins))

        reward = 1 if np.abs(new_angle) < 12 * np.pi / 180 else -1
        done = True if np.abs(new_angle) >= 12 * np.pi / 180 or np.abs(new_position) > 2.4 else False

        return new_state_idx, reward, done
