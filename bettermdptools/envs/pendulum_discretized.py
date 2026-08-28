"""
Author: Aleksandr Spiridonov
BSD 3-Clause License
"""

import gzip
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_start_method
from numbers import Integral
from tempfile import NamedTemporaryFile

import numpy as np

from bettermdptools.envs.binning import generate_bin_edges

CACHE_VERSION = 2
CACHED_P_PATH_FORMAT = (
    "cached_P_discretized_pendulum_v{cache_version}_{angle_bins}_"
    "{angular_velocity_bins}_{action_bins}_{dim_samples}.pkl.gz"
)


def _can_start_worker_processes():
    """Return whether the current main module can be imported by workers."""
    if get_start_method() == "fork":
        return True

    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    return main_file is not None and os.path.exists(main_file)


def _is_complete_transition_model(model, state_space, action_space):
    """Return whether a model contains every expected state and action."""
    if not isinstance(model, dict) or set(model) != set(range(state_space)):
        return False

    expected_actions = set(range(action_space))
    return all(
        isinstance(model[state], dict)
        and set(model[state]) == expected_actions
        and all(model[state][action] for action in expected_actions)
        for state in range(state_space)
    )


def _write_cached_transition_model(model, cached_model_path):
    """Atomically publish a completed transition model."""
    cache_dir = os.path.dirname(cached_model_path)
    with NamedTemporaryFile(dir=cache_dir, delete=False) as temporary_file:
        temporary_path = temporary_file.name

    try:
        with gzip.open(temporary_path, "wb") as file:
            pickle.dump(model, file)
        os.replace(temporary_path, cached_model_path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def angle_normalize(angle):
    """Normalize an angle to the half-open interval [-pi, pi)."""
    normalized = np.remainder(angle, 2 * np.pi)
    if np.ndim(normalized) == 0:
        return normalized - 2 * np.pi if normalized >= np.pi else normalized
    return np.where(normalized >= np.pi, normalized - 2 * np.pi, normalized)


def wrap(value, lower_bound, upper_bound):
    """Wrap a scalar into the half-open interval [lower_bound, upper_bound)."""
    if (
        not np.isfinite(value)
        or not np.isfinite(lower_bound)
        or not np.isfinite(upper_bound)
        or lower_bound >= upper_bound
    ):
        raise ValueError("wrap requires finite bounds with lower_bound < upper_bound")
    difference = upper_bound - lower_bound
    while value >= upper_bound:
        value -= difference
    while value < lower_bound:
        value += difference
    return value


def _validate_index(name, value, size):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [0, {size})")
    value = int(value)
    if not 0 <= value < size:
        raise ValueError(f"{name} must be an integer in [0, {size})")
    return value


def _validate_sample_count(num_samples):
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, Integral)
        or num_samples < 3
    ):
        raise ValueError("num_samples must be an integer greater than or equal to 3")
    return int(num_samples)


def index_to_state(index, angle_bins, angular_velocity_bins):
    """Convert a flat state index into angle and velocity bin indices."""
    index = _validate_index("index", index, angle_bins * angular_velocity_bins)
    angle_idx = index // angular_velocity_bins
    angular_velocity_idx = index % angular_velocity_bins
    return angle_idx, angular_velocity_idx


def index_to_continuous_state(index, angle_bin_edges, angular_velocity_bin_edges):
    """Return midpoint angle and velocity values for a flat state index."""
    angle_idx, angular_velocity_idx = index_to_state(
        index, len(angle_bin_edges) - 1, len(angular_velocity_bin_edges) - 1
    )
    angle = (angle_bin_edges[angle_idx] + angle_bin_edges[angle_idx + 1]) / 2.0
    angular_velocity = (
        angular_velocity_bin_edges[angular_velocity_idx]
        + angular_velocity_bin_edges[angular_velocity_idx + 1]
    ) / 2.0
    return angle, angular_velocity


def index_to_continous_state(index, angle_bin_edges, angular_velocity_bin_edges):
    """Compatibility alias for the historically misspelled public function."""
    return index_to_continuous_state(
        index,
        angle_bin_edges,
        angular_velocity_bin_edges,
    )


def state_to_index(angle_idx, angular_velocity_idx, angular_velocity_bins):
    """Convert angle and velocity bin indices into a flat state index."""
    if (
        isinstance(angular_velocity_bins, bool)
        or not isinstance(angular_velocity_bins, Integral)
        or angular_velocity_bins < 1
    ):
        raise ValueError("angular_velocity_bins must be a positive integer")
    if (
        isinstance(angle_idx, bool)
        or not isinstance(angle_idx, Integral)
        or angle_idx < 0
    ):
        raise ValueError("angle_idx must be a nonnegative integer")
    angular_velocity_idx = _validate_index(
        "angular_velocity_idx", angular_velocity_idx, angular_velocity_bins
    )
    return angle_idx * angular_velocity_bins + angular_velocity_idx


def get_torque_value(torque_bin_edges, action):
    """Return the midpoint torque represented by a discrete action."""
    action = _validate_index("action", action, len(torque_bin_edges) - 1)
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
    l=1.0,  # noqa: E741 - Preserve the existing public keyword argument.
    m=1.0,
    dt=0.05,
):
    """Approximate one-step transitions by sampling within a state bin."""
    num_samples = _validate_sample_count(num_samples)
    angle_idx = _validate_index("angle_idx", angle_idx, len(angle_bin_edges) - 1)
    angular_velocity_idx = _validate_index(
        "angular_velocity_idx",
        angular_velocity_idx,
        len(angular_velocity_bin_edges) - 1,
    )
    angle_low, angle_high = angle_bin_edges[angle_idx], angle_bin_edges[angle_idx + 1]
    angular_velocity_low, angular_velocity_high = (
        angular_velocity_bin_edges[angular_velocity_idx],
        angular_velocity_bin_edges[angular_velocity_idx + 1],
    )
    torque = get_torque_value(torque_bin_edges, action)

    min_angular_velocity = angular_velocity_bin_edges[0]
    max_angular_velocity = angular_velocity_bin_edges[-1]

    angle_samples = np.linspace(angle_low, angle_high, num_samples)
    angle_samples = angle_samples[1:-1]  # Exclude the bin edges
    angular_velocity_samples = np.linspace(
        angular_velocity_low, angular_velocity_high, num_samples
    )
    angular_velocity_samples = angular_velocity_samples[1:-1]

    angle_bins = len(angle_bin_edges) - 1
    angular_velocity_bins = len(angular_velocity_bin_edges) - 1

    next_states_and_rewards = {}

    for angle in angle_samples:
        for angular_velocity in angular_velocity_samples:
            costs = (
                angle_normalize(angle) ** 2
                + 0.1 * angular_velocity**2
                + 0.001 * (torque**2)
            )

            new_angular_velocity = (
                angular_velocity
                + (3 * g / (2 * l) * np.sin(angle) + 3.0 / (m * l**2) * torque) * dt
            )
            new_angular_velocity = np.clip(
                new_angular_velocity,
                min_angular_velocity + 1e-6,
                max_angular_velocity - 1e-6,
            )

            new_angle = angle_normalize(angle + new_angular_velocity * dt)

            new_angle_idx = np.clip(
                np.digitize(new_angle, angle_bin_edges) - 1,
                0,
                angle_bins - 1,
            )
            new_angular_velocity_idx = np.clip(
                np.digitize(new_angular_velocity, angular_velocity_bin_edges) - 1,
                0,
                angular_velocity_bins - 1,
            )

            new_state = state_to_index(
                new_angle_idx, new_angular_velocity_idx, angular_velocity_bins
            )

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
    """Build all action transitions for one state from serialized arguments."""
    (
        state,
        angle_bin_edges,
        angular_velocity_bin_edges,
        torque_bin_edges,
        dim_samples,
    ) = args
    angle_bins = len(angle_bin_edges) - 1
    angular_velocity_bins = len(angular_velocity_bin_edges) - 1
    action_bins = len(torque_bin_edges) - 1

    P_state = {action: [] for action in range(action_bins)}

    angle_idx, angular_velocity_idx = index_to_state(
        state, angle_bins, angular_velocity_bins
    )

    for action in range(action_bins):
        P_state[action] = compute_next_probable_states(
            angle_idx,
            angular_velocity_idx,
            action,
            angle_bin_edges,
            angular_velocity_bin_edges,
            torque_bin_edges,
            num_samples=dim_samples,
        )

    return state, P_state


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
        Positive number of worker processes used to build transition probabilities.
    cache_dir : str, optional (default='./cached')
        Directory to cache the transition probabilities.
    dim_samples : int, optional (default=11)
        Samples per dimension when setting up transition probabilities. Must be
        an integer of at least 3 and is part of the model cache identity.
    Attributes:
    -----------
    angle_bins : int
        Number of bins to discretize the angle. Must be odd.
    angular_velocity_bins : int
        Number of bins to discretize the angular velocity. Must be odd.
    dim_samples : int
        Samples per dimension when setting up transition probabilities.
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
        cache_dir="./cached",
        dim_samples=11,
    ):
        dim_samples = _validate_sample_count(dim_samples)
        if (
            isinstance(n_workers, bool)
            or not isinstance(n_workers, Integral)
            or n_workers < 1
        ):
            raise ValueError("n_workers must be a positive integer")

        self.angle_bins = angle_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.dim_samples = dim_samples
        self.angle_bin_edges = generate_bin_edges(np.pi, angle_bins, 3, center=True)
        self.angular_velocity_bin_edges = generate_bin_edges(
            8, angular_velocity_bins, 3, center=False
        )
        self.torque_bin_edges = generate_bin_edges(2, torque_bins, 3, center=False)

        self.state_space = angle_bins * angular_velocity_bins
        self.action_space = torque_bins

        self.P = {
            state: {action: [] for action in range(torque_bins)}
            for state in range(self.state_space)
        }

        self.n_workers = int(n_workers)

        cached_P_filepath = CACHED_P_PATH_FORMAT.format(
            cache_version=CACHE_VERSION,
            angle_bins=angle_bins,
            angular_velocity_bins=angular_velocity_bins,
            action_bins=torque_bins,
            dim_samples=dim_samples,
        )
        cached_P_filepath = os.path.join(cache_dir, cached_P_filepath)

        os.makedirs(cache_dir, exist_ok=True)

        cached_P = None
        if os.path.exists(cached_P_filepath):
            try:
                with gzip.open(cached_P_filepath, "rb") as file:
                    cached_P = pickle.load(file)
            except (EOFError, OSError, pickle.PickleError):
                pass

        if _is_complete_transition_model(cached_P, self.state_space, self.action_space):
            self.P = cached_P
        else:
            self.setup_transition_probabilities()
            if not _is_complete_transition_model(
                self.P, self.state_space, self.action_space
            ):
                raise RuntimeError(
                    "Pendulum transition model construction was incomplete"
                )
            _write_cached_transition_model(self.P, cached_P_filepath)

    def discretize_angle(self, angle):
        """Map a finite angle to its discrete bin index."""
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        angle = angle_normalize(angle)
        return int(
            np.clip(
                np.digitize(angle, self.angle_bin_edges) - 1,
                0,
                self.angle_bins - 1,
            )
        )

    def discretize_angular_velocity(self, angular_velocity):
        """Map a finite angular velocity to its discrete bin index."""
        if not np.isfinite(angular_velocity):
            raise ValueError("angular_velocity must be finite")
        return int(
            np.clip(
                np.digitize(angular_velocity, self.angular_velocity_bin_edges) - 1,
                0,
                self.angular_velocity_bins - 1,
            )
        )

    def index_to_state(self, index):
        """Convert a flat state index into angle and velocity bin indices."""
        return index_to_state(index, self.angle_bins, self.angular_velocity_bins)

    def state_to_index(self, angle_idx, angular_velocity_idx):
        """Convert angle and velocity bin indices into a flat state index."""
        angle_idx = _validate_index("angle_idx", angle_idx, self.angle_bins)
        angular_velocity_idx = _validate_index(
            "angular_velocity_idx",
            angular_velocity_idx,
            self.angular_velocity_bins,
        )
        idx = state_to_index(
            angle_idx, angular_velocity_idx, self.angular_velocity_bins
        )
        return int(idx)

    def transform_cont_obs(self, cont_obs):
        """Convert a Gymnasium Pendulum observation to a state index."""
        cont_obs = np.asarray(cont_obs)
        if cont_obs.shape != (3,) or not np.isfinite(cont_obs).all():
            raise ValueError("Pendulum observations must contain three finite values")
        x = cont_obs[0]
        y = cont_obs[1]
        theta = angle_normalize(np.arctan2(y, x))
        theta_dot = cont_obs[2]
        theta_dot = np.clip(theta_dot, -8 + 1e-6, 8 - 1e-6)

        angle_idx = self.discretize_angle(theta)
        angular_velocity_idx = self.discretize_angular_velocity(theta_dot)

        return self.state_to_index(angle_idx, angular_velocity_idx)

    def get_action_value(self, action):
        """Return the midpoint torque represented by a discrete action."""
        return get_torque_value(self.torque_bin_edges, action)

    def setup_transition_probabilities(self):
        """Build the sampled transition model serially or with worker processes."""
        state_space_values = list(range(self.state_space))

        args = [
            (
                state,
                self.angle_bin_edges,
                self.angular_velocity_bin_edges,
                self.torque_bin_edges,
                self.dim_samples,
            )
            for state in state_space_values
        ]

        new_P = {}

        args = [arg for arg in args if arg[0] not in new_P]

        num_workers = self.n_workers if _can_start_worker_processes() else 1

        n_completed = len(new_P)

        batch_size = 1000

        if num_workers == 1:
            for arg in args:
                state, P_state = setup_transition_probabilities_for_state(arg)
                new_P[state] = P_state
            self.P = new_P
            return

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(args), batch_size):
                batch = args[i : i + batch_size]
                futures = [
                    executor.submit(setup_transition_probabilities_for_state, arg)
                    for arg in batch
                ]
                for future in as_completed(futures):
                    n_completed += 1
                    state, P_state = future.result()
                    new_P[state] = P_state
                    if n_completed % 100 == 0:
                        print(f"Completed {n_completed}/{self.state_space}")

        self.P = new_P


if __name__ == "__main__":
    n_bins = 31
    angle_bins = n_bins
    angular_velocity_bins = n_bins

    discretized_pendulum = DiscretizedPendulum(
        angle_bins=angle_bins, angular_velocity_bins=angular_velocity_bins
    )

    angle = np.pi / 2
    angular_velocity = 3

    obs = np.array([np.cos(angle), np.sin(angle), angular_velocity])

    state = discretized_pendulum.transform_cont_obs(obs)
    print(f"Discretized state index: {state}")

    for action in range(discretized_pendulum.action_space):
        transitions = discretized_pendulum.P[state][action]
        for prob, next_state, reward, terminated in transitions:
            print(
                f"Action: {action}, Probability: {prob}, "
                f"Next state: {next_state}, Reward: {reward}, "
                f"Terminated: {terminated}"
            )
