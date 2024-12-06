"""
Author: Aleksandr Spiridonov
BSD 3-Clause License
"""
import numpy as np
from bettermdptools.envs.binning import generate_bin_edges
from gymnasium.envs.classic_control.pendulum import angle_normalize
from gymnasium.envs.classic_control.acrobot import wrap
import os
import gzip
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


CACHED_P_PATH_FORMAT = 'cached_P_discretized_pendulum_{angle_bins}_{angular_velocity_bins}_{action_bins}.pkl.gz'

def index_to_state(
        index,
        angle_bins,
        angular_velocity_bins
):
    angle_idx = index // angular_velocity_bins
    angular_velocity_idx = index % angular_velocity_bins
    return angle_idx, angular_velocity_idx

def index_to_continous_state(
        index,
        angle_bin_edges,
        angular_velocity_bin_edges
):
    angle_idx, angular_velocity_idx = index_to_state(index, len(angle_bin_edges) - 1, len(angular_velocity_bin_edges) - 1)
    angle = (angle_bin_edges[angle_idx] + angle_bin_edges[angle_idx + 1]) / 2.0
    angular_velocity = (angular_velocity_bin_edges[angular_velocity_idx] + angular_velocity_bin_edges[angular_velocity_idx + 1]) / 2.0
    return angle, angular_velocity

def state_to_index(
        angle_idx,
        angular_velocity_idx,
        angular_velocity_bins
):
    return angle_idx * angular_velocity_bins + angular_velocity_idx

def get_torque_value(torque_bin_edges, action):
    return (torque_bin_edges[action] + torque_bin_edges[action + 1]) / 2.0

def compute_next_probable_states(
        angle_idx,
        angular_velocity_idx,
        action,
        angle_bin_edges,
        angular_velocity_bin_edges,
        torque_bin_edges,
        num_samples=11,
        g=10.0,
        l=1.0,
        m=1.0,
        dt=0.05
):
    angle_low, angle_high = angle_bin_edges[angle_idx], angle_bin_edges[angle_idx + 1]
    angular_velocity_low, angular_velocity_high = angular_velocity_bin_edges[angular_velocity_idx], angular_velocity_bin_edges[angular_velocity_idx + 1]
    torque = get_torque_value(torque_bin_edges, action)

    min_angular_velocity = angular_velocity_bin_edges[0]
    max_angular_velocity = angular_velocity_bin_edges[-1]

    angle_samples = np.linspace(angle_low, angle_high, num_samples)
    angle_samples = angle_samples[1:-1] # Exclude the bin edges
    angular_velocity_samples = np.linspace(angular_velocity_low, angular_velocity_high, num_samples)
    angular_velocity_samples = angular_velocity_samples[1:-1]

    angle_bins = len(angle_bin_edges) - 1
    angular_velocity_bins = len(angular_velocity_bin_edges) - 1

    next_states_and_rewards = {}

    for angle in angle_samples:
        for angular_velocity in angular_velocity_samples:
            costs = angle_normalize(angle) ** 2 + 0.1 * angular_velocity ** 2 + 0.001 * (torque ** 2)

            new_angular_velocity = angular_velocity + (3 * g / (2 * l) * np.sin(angle) + 3.0 / (m * l ** 2) * torque) * dt
            new_angular_velocity = np.clip(new_angular_velocity, min_angular_velocity+1e-6, max_angular_velocity-1e-6)

            new_angle = angle + new_angular_velocity * dt
            new_angle = wrap(new_angle, -np.pi, np.pi)

            new_angle_idx = np.digitize(new_angle, angle_bin_edges) - 1
            new_angular_velocity_idx = np.digitize(new_angular_velocity, angular_velocity_bin_edges) - 1

            new_state = state_to_index(
                new_angle_idx, 
                new_angular_velocity_idx, 
                angular_velocity_bins)
            
            if new_state < 0 or new_state >= angle_bins * angular_velocity_bins:
                raise ValueError(f"Invalid state index: {new_state}")
            
            terminated = False

            summary = (new_state, -costs, terminated)

            if new_state not in next_states_and_rewards:
                next_states_and_rewards[new_state] = []
            next_states_and_rewards[new_state].append(summary)

    n_total = len(angle_samples) * len(angular_velocity_samples)

    results = []

    for new_state, summaries in next_states_and_rewards.items():
        n = len(summaries)
        prob = n / n_total
        ave_reward = sum(r for _, r, _ in summaries) / n
        terminated = False
        results.append((prob, new_state, ave_reward, terminated))

    return results

def setup_transition_probabilities_for_state(args):
    state, angle_bin_edges, angular_velocity_bin_edges, torque_bin_edges, dim_samples = args
    angle_bins = len(angle_bin_edges) - 1
    angular_velocity_bins = len(angular_velocity_bin_edges) - 1
    action_bins = len(torque_bin_edges) - 1

    P_state = {action: [] for action in range(action_bins)}

    angle_idx, angular_velocity_idx = index_to_state(state, angle_bins, angular_velocity_bins)

    for action in range(action_bins):
        P_state[action] = compute_next_probable_states(
            angle_idx,
            angular_velocity_idx,
            action,
            angle_bin_edges,
            angular_velocity_bin_edges,
            torque_bin_edges,
            num_samples=dim_samples
        )

    try:
        return (state, P_state)
    except Exception as e:
        print(f"Error in state {state}: {e}")
        return None

class DiscretizedPendulum:
    """
    Initialize the DiscretizedPendulum environment.
    Parameters:
    -----------
    angle_bins : int
        Number of bins to discretize the angle.
    angular_velocity_bins : int
        Number of bins to discretize the angular velocity.
    torque_bins : int, optional (default=11)
        Number of bins to discretize the torque.
    n_workers : int, optional (default=4)
        Number of worker processes to use for setting up transition probabilities.
    cache_dir : str, optional (default='./cached')
        Directory to cache the transition probabilities.
    dim_samples : int, optional (default=11)
        Number of samples to use for each dimension when setting up transition probabilities.
    Attributes:
    -----------
    angle_bins : int
        Number of bins to discretize the angle. Must be odd.
    angular_velocity_bins : int
        Number of bins to discretize the angular velocity. Must be odd.
    dim_samples : int
        Number of samples to use for each dimension when setting up transition probabilities.
    angle_bin_edges : numpy.ndarray
        Edges of the bins for discretizing the angle.
    angular_velocity_bin_edges : numpy.ndarray
        Edges of the bins for discretizing the angular velocity.
    torque_bin_edges : numpy.ndarray
        Edges of the bins for discretizing the torque.
    state_space : int
        Total number of discrete states.
    action_space : int
        Total number of discrete actions.
    P : dict
        Transition probability matrix.
    n_workers : int
        Number of worker processes to use for setting up transition probabilities.
    """
    def __init__(
            self,
            angle_bins,
            angular_velocity_bins,
            torque_bins=11,
            n_workers=4,
            cache_dir='./cached',
            dim_samples=11
    ):
        self.angle_bins = angle_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.dim_samples = dim_samples
        self.angle_bin_edges = generate_bin_edges(np.pi, angle_bins, 3, center=True)
        self.angular_velocity_bin_edges = generate_bin_edges(8, angular_velocity_bins, 3, center=False)
        self.torque_bin_edges = generate_bin_edges(2, torque_bins, 3, center=False)

        self.state_space = angle_bins * angular_velocity_bins
        self.action_space = torque_bins

        self.P = {state: {action: [] for action in range(torque_bins)} for state in range(self.state_space)}

        self.n_workers = n_workers

        cached_P_filepath = CACHED_P_PATH_FORMAT.format(angle_bins=angle_bins, angular_velocity_bins=angular_velocity_bins, action_bins=torque_bins)
        cached_P_filepath = os.path.join(cache_dir, cached_P_filepath)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if os.path.exists(cached_P_filepath):
            with gzip.open(cached_P_filepath, 'rb') as f:
                self.P = pickle.load(f)
        else:
            self.setup_transition_probabilities()
            with gzip.open(cached_P_filepath, 'wb') as f:
                pickle.dump(self.P, f)

    def discretize_angle(self, angle):
        return np.digitize(angle, self.angle_bin_edges) - 1
    
    def discretize_angular_velocity(self, angular_velocity):
        return np.digitize(angular_velocity, self.angular_velocity_bin_edges) - 1
    
    def index_to_state(self, index):
        return index_to_state(index, self.angle_bins, self.angular_velocity_bins)
    
    def state_to_index(self, angle_idx, angular_velocity_idx):
        idx = state_to_index(angle_idx, angular_velocity_idx, self.angular_velocity_bins)
        if idx < 0 or idx >= self.state_space:
            raise ValueError(f"Invalid state index: {idx}")
        return idx

    def transform_cont_obs(self, cont_obs):
        x = cont_obs[0]
        y = cont_obs[1]
        theta = np.arctan2(y, x)
        theta = wrap(theta, -np.pi, np.pi)
        theta_dot = cont_obs[2]
        theta_dot = np.clip(theta_dot, -8+1e-6, 8-1e-6)


        angle_idx = self.discretize_angle(theta)
        angular_velocity_idx = self.discretize_angular_velocity(theta_dot)

        return self.state_to_index(angle_idx, angular_velocity_idx)

    def get_action_value(self, action):
        return get_torque_value(self.torque_bin_edges, action)

    def setup_transition_probabilities(self):
        state_space_values = list(range(self.state_space))

        args = [
            (
                state,
                self.angle_bin_edges,
                self.angular_velocity_bin_edges,
                self.torque_bin_edges,
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
                batch = args[i:i + batch_size]
                futures = [executor.submit(setup_transition_probabilities_for_state, arg) for arg in batch]
                for future in as_completed(futures):
                    n_completed += 1
                    try:
                        state, P_state = future.result()
                        new_P[state] = P_state
                        if n_completed % 100 == 0:
                            print(f'Completed {n_completed}/{self.state_space}')
                    except Exception as e:
                        print(f"Error in future: {e}")
                        print('task failed')

        self.P = new_P
        
if __name__ == '__main__':
    n_bins = 31
    angle_bins = n_bins
    angular_velocity_bins = n_bins

    discretized_pendulum = DiscretizedPendulum(
        angle_bins=angle_bins,
        angular_velocity_bins=angular_velocity_bins
    )

    angle = np.pi / 2
    angular_velocity = 3

    obs = np.array([np.cos(angle), np.sin(angle), angular_velocity])

    state = discretized_pendulum.transform_cont_obs(obs)
    print(f'Discretized state index: {state}')

    for action in range(discretized_pendulum.action_space):
        transitions = discretized_pendulum.P[state][action]
        for prob, next_state, reward, terminated in transitions:
            print(f'Action: {action}, Probability: {prob}, Next state: {next_state}, Reward: {reward}, Terminated: {terminated}')