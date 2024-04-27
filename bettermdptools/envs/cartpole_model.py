"""
generative AI experiment - discretized cartpole transition and reward (P) matrix with adaptive angle binning
created with chatGPT

Example usage:
dpole = DiscretizedCartPole(5, 5, 5, .001, .1)  # Example bin sizes for each variable and adaptive angle binning center/outer resolution

"""

import numpy as np


class DiscretizedCartPole:
    def __init__(self, position_bins, velocity_bins, angular_velocity_bins, angular_center_resolution, angular_outer_resolution):
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
        self.convert_state_lambda = lambda obs: (
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
        for state in range(self.state_space):
            position, velocity, angle, angular_velocity = self.index_to_state(state)
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(position, velocity, angle, angular_velocity, action)
                self.P[state][action].append((1, next_state, reward, done))


    def index_to_state(self, index):
        totals = [self.position_bins, self.velocity_bins, len(self.angle_bins), self.angular_velocity_bins]
        multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        components = [int((index // multipliers[i]) % totals[i]) for i in range(4)]
        return components

    def compute_next_state(self, position_idx, velocity_idx, angle_idx, angular_velocity_idx, action):
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
