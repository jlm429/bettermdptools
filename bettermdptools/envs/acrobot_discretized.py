"""
Author: Aleksandr Spiridonov
BSD 3-Clause License
"""
import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, rk4, wrap, bound
import os
import gzip
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

CACHED_P_PATH_FORMAT = 'cached_P_discretized_acrobot_{angle_bins}_{angular_velocity_bins}.pkl.gz'

def _dsdt(s_augmented):
    m1 = AcrobotEnv.LINK_MASS_1
    m2 = AcrobotEnv.LINK_MASS_2
    l1 = AcrobotEnv.LINK_LENGTH_1
    lc1 = AcrobotEnv.LINK_COM_POS_1
    lc2 = AcrobotEnv.LINK_COM_POS_2
    I1 = AcrobotEnv.LINK_MOI
    I2 = AcrobotEnv.LINK_MOI
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

def index_to_state(
        index,
        angle_bins,
        angular_velocity_bins
        ):
    angle1_idx = index // (angle_bins * angular_velocity_bins**2)
    angle2_idx = (index // (angular_velocity_bins**2)) % angle_bins
    angular_velocity1_idx = (index // angular_velocity_bins) % angular_velocity_bins
    angular_velocity2_idx = index % angular_velocity_bins

    return angle1_idx, angle2_idx, angular_velocity1_idx, angular_velocity2_idx

def index_to_continuous_state(
        index,
        angle_bin_edges,
        angular_velocity_bin_edges_1,
        angular_velocity_bin_edges_2
        ):
    angle_bins = len(angle_bin_edges) - 1
    angular_velocity_bins = len(angular_velocity_bin_edges_1) - 1
    angle1_idx, angle2_idx, angular_velocity1_idx, angular_velocity2_idx = index_to_state(
        index,
        angle_bins,
        angular_velocity_bins
    )

    angle1 = (angle_bin_edges[angle1_idx] + angle_bin_edges[angle1_idx + 1] ) / 2
    angle2 = (angle_bin_edges[angle2_idx] + angle_bin_edges[angle2_idx + 1] ) / 2
    angular_velocity1 = (angular_velocity_bin_edges_1[angular_velocity1_idx] + angular_velocity_bin_edges_1[angular_velocity1_idx + 1] ) / 2
    angular_velocity2 = (angular_velocity_bin_edges_2[angular_velocity2_idx] + angular_velocity_bin_edges_2[angular_velocity2_idx + 1] ) / 2

    return angle1, angle2, angular_velocity1, angular_velocity2


def state_to_index(
        angle1_idx, 
        angle2_idx, 
        angular_velocity1_idx, 
        angular_velocity2_idx,
        angle_bins,
        angular_velocity_bins
        ):
        return angle1_idx * (angle_bins * angular_velocity_bins**2) + angle2_idx * (angular_velocity_bins**2) + angular_velocity1_idx * angular_velocity_bins + angular_velocity2_idx

def compute_next_probable_states(
        angle1_idx, 
        angle2_idx, 
        angular_velocity1_idx, 
        angular_velocity2_idx, 
        action, 
        angle_bin_edges,
        angular_velocity_bin_edges_1,
        angular_velocity_bin_edges_2,
        num_samples=6):
    angle1_low, angle1_high = angle_bin_edges[angle1_idx:angle1_idx + 2]
    angle2_low, angle2_high = angle_bin_edges[angle2_idx:angle2_idx + 2]
    angular_velocity1_low, angular_velocity1_high = angular_velocity_bin_edges_1[angular_velocity1_idx:angular_velocity1_idx + 2]
    angular_velocity2_low, angular_velocity2_high = angular_velocity_bin_edges_2[angular_velocity2_idx:angular_velocity2_idx + 2]

    angle1_samples = np.linspace(angle1_low, angle1_high, num_samples)
    angle1_samples = angle1_samples[1:-1]  # Remove the edges
    angle2_samples = np.linspace(angle2_low, angle2_high, num_samples)
    angle2_samples = angle2_samples[1:-1]
    angular_velocity1_samples = np.linspace(angular_velocity1_low, angular_velocity1_high, num_samples)
    angular_velocity1_samples = angular_velocity1_samples[1:-1]
    angular_velocity2_samples = np.linspace(angular_velocity2_low, angular_velocity2_high, num_samples)
    angular_velocity2_samples = angular_velocity2_samples[1:-1]

    angle_bins = len(angle_bin_edges) - 1
    angular_velocity_bins = len(angular_velocity_bin_edges_1) - 1

    # terminal condition = bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)

    next_states_and_rewards = {}
    torque = AcrobotEnv.AVAIL_TORQUE[action]

    for angle1 in angle1_samples:
        for angle2 in angle2_samples:
            for angular_velocity1 in angular_velocity1_samples:
                for angular_velocity2 in angular_velocity2_samples:
                    s = np.array([angle1, angle2, angular_velocity1, angular_velocity2])
                    s_augmented = np.append(s, torque)
                    ns = rk4(_dsdt, s_augmented, [0, AcrobotEnv.dt])

                    ns[0] = wrap(ns[0], -np.pi, np.pi)
                    ns[1] = wrap(ns[1], -np.pi, np.pi)
                    ns[2] = bound(ns[2], -AcrobotEnv.MAX_VEL_1+1e-6, AcrobotEnv.MAX_VEL_1-1e-6)
                    ns[3] = bound(ns[3], -AcrobotEnv.MAX_VEL_2+1e-6, AcrobotEnv.MAX_VEL_2-1e-6)

                    new_angle1_idx = np.digitize(ns[0], angle_bin_edges) - 1
                    new_angle2_idx = np.digitize(ns[1], angle_bin_edges) - 1
                    new_angular_velocity1_idx = np.digitize(ns[2], angular_velocity_bin_edges_1) - 1
                    new_angular_velocity2_idx = np.digitize(ns[3], angular_velocity_bin_edges_2) - 1

                    new_state = state_to_index(
                        new_angle1_idx, 
                        new_angle2_idx, 
                        new_angular_velocity1_idx, 
                        new_angular_velocity2_idx,
                        angle_bins,
                        angular_velocity_bins)
                    
                    if new_state < 0 or new_state >= angle_bins * angle_bins * angular_velocity_bins * angular_velocity_bins:
                        raise ValueError(f'Invalid state: {new_state}')

                    terminated = bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.0)
                    reward = -1.0 if not terminated else 0.0

                    summary = (new_state, reward, terminated)

                    if new_state not in next_states_and_rewards:
                        next_states_and_rewards[new_state] = []
                    next_states_and_rewards[new_state].append(summary)

    n_total = len(angle1_samples) * len(angle2_samples) * len(angular_velocity1_samples) * len(angular_velocity2_samples)

    results = []
    for new_state, summaries in next_states_and_rewards.items():
        n = len(summaries)
        p = n / n_total
        ave_reward = sum([summary[1] for summary in summaries]) / n
        terminated = all([summary[2] for summary in summaries])
        results.append((p, new_state, ave_reward, terminated))

    return results

def setup_transition_probabilities_for_state(args):
    state, angle_bin_edges, angular_velocity_bin_edges_1, angular_velocity_bin_edges_2, angle_bins, angular_velocity_bins, action_space, dim_samples = args

    P_state = {action: [] for action in range(action_space)}

    angle1_idx, angle2_idx, angular_velocity1_idx, angular_velocity2_idx = index_to_state(
        state, 
        angle_bins, 
        angular_velocity_bins
    )

    for action in range(action_space):
        P_state[action] = compute_next_probable_states(
            angle1_idx, 
            angle2_idx, 
            angular_velocity1_idx, 
            angular_velocity2_idx, 
            action,
            angle_bin_edges,
            angular_velocity_bin_edges_1,
            angular_velocity_bin_edges_2,
            num_samples=dim_samples
        )

    # print('thread done')
    try:
        # Existing code...
        return (state, P_state)
    except Exception as e:
        print(f"Error in state {args[0]}: {e}")
        return None

class DiscretizedAcrobot:
    def __init__(
            self,
            angle_bins,
            angular_velocity_bins,
            n_workers=4,
            cache_dir='./cached',
            dim_samples=6
    ):
        self.angle_bins = angle_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.n_workers = n_workers
        self.dim_samples = dim_samples

        self.angle_bin_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)
        self.angular_velocity_bin_edges_1 = np.linspace(-4 * np.pi, 4 * np.pi, angular_velocity_bins + 1)
        self.angular_velocity_bin_edges_2 = np.linspace(-9 * np.pi, 9 * np.pi, angular_velocity_bins + 1)

        self.state_space = self.angle_bins**2 * self.angular_velocity_bins**2
        self.action_space = 3

        self.P = {state: {action: [] for action in range(self.action_space)} for state in range(self.state_space)}

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        cached_filepath = CACHED_P_PATH_FORMAT.format(angle_bins=angle_bins, angular_velocity_bins=angular_velocity_bins)
        cached_filepath = os.path.join(cache_dir, cached_filepath)

        if os.path.exists(cached_filepath):
            with gzip.open(cached_filepath, 'rb') as f:
                self.P = pickle.load(f)
        else:
            self.setup_transition_probabilities()
            with gzip.open(cached_filepath, 'wb') as f:
                pickle.dump(self.P, f)




    def discretize_angle1(self, angle1):
        return np.digitize(angle1, self.angle_bin_edges) - 1
    
    def discretize_angle2(self, angle2):
        return np.digitize(angle2, self.angle_bin_edges) - 1
    
    def discretize_angular_velocity1(self, angular_velocity1):
        return np.digitize(angular_velocity1, self.angular_velocity_bin_edges_1) - 1
    
    def discretize_angular_velocity2(self, angular_velocity2):
        return np.digitize(angular_velocity2, self.angular_velocity_bin_edges_2) - 1
    
    # continuous observation space consists of 
    # cosine of angle 1 [-1, 1]
    # sine of angle 1 [-1, 1]
    # cosine of angle 2 [-1, 1]
    # sine of angle 2 [-1, 1]
    # angular velocity 1 [-4 * pi, 4 * pi]
    # angular velocity 2 [-9 * pi, 9 * pi]
    def transform_cont_obs(self, cont_obs):
        angle1 = np.arctan2(cont_obs[1], cont_obs[0])
        angle2 = np.arctan2(cont_obs[3], cont_obs[2])
        angular_velocity1 = cont_obs[4]
        angular_velocity2 = cont_obs[5]

        angle1_idx = self.discretize_angle1(angle1)
        angle2_idx = self.discretize_angle2(angle2)
        angular_velocity1_idx = self.discretize_angular_velocity1(angular_velocity1)
        angular_velocity2_idx = self.discretize_angular_velocity2(angular_velocity2)

        return self.state_to_index(angle1_idx, angle2_idx, angular_velocity1_idx, angular_velocity2_idx)

    def index_to_state(self, index):
        return index_to_state(
            index,
            self.angle_bins,
            self.angular_velocity_bins
        )

    def state_to_index(self, angle1_idx, angle2_idx, angular_velocity1_idx, angular_velocity2_idx):
        return state_to_index(
            angle1_idx, 
            angle2_idx, 
            angular_velocity1_idx, 
            angular_velocity2_idx,
            self.angle_bins,
            self.angular_velocity_bins
        )

    def setup_transition_probabilities(self):
        state_space_values = list(range(self.state_space))
        state_space_values.reverse()
        args = [
            (
                state,
                self.angle_bin_edges,
                self.angular_velocity_bin_edges_1,
                self.angular_velocity_bin_edges_2,
                self.angle_bins,
                self.angular_velocity_bins,
                self.action_space,
                self.dim_samples
            )
            for state in state_space_values
        ]

        new_P = {}

        args = [arg for arg in args if arg[0] not in new_P]

        num_workers = self.n_workers

        n_completed = len(new_P)

        batch_size = 1000

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(args), batch_size):
                batch = args[i:i+batch_size]
                futures = [executor.submit(setup_transition_probabilities_for_state, arg) for arg in batch]
                for future in as_completed(futures):
                    n_completed += 1
                    try:
                        state, P_state = future.result()
                        new_P[state] = P_state
                        if n_completed % 100 == 0:
                            print(f'{n_completed} states completed out of {self.state_space}')
                    except Exception as e:
                        print(f'Error: {e}')
                        print('task failed')

        self.P = new_P


if __name__ == '__main__':
    angle_bins = 11
    angular_velocity_bins = 11
    
    import gymnasium as gym


    discretized_acrobot = DiscretizedAcrobot(
        angle_bins=angle_bins,
        angular_velocity_bins=angular_velocity_bins,
        n_workers=20
    )

    angle1 = np.pi / 3
    angle2 = np.pi / 4
    angular_velocity1 = 0.1
    angular_velocity2 = 0.2

    obs = [np.cos(angle1), np.sin(angle1), np.cos(angle2), np.sin(angle2), angular_velocity1, angular_velocity2]

    state = discretized_acrobot.transform_cont_obs(obs)
    print(f'Discrete state index: {state}')

    for action in range(discretized_acrobot.action_space):
        transitions = discretized_acrobot.P[state][action]
        for prob, next_state, reward, terminated in transitions:
            print(f'Action: {action}, Probability: {prob}, Next state: {next_state}, Reward: {reward}, Terminated: {terminated}')

    






