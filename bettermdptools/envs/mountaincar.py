import gymnasium as gym
import pandas as pd
from gymnasium.utils import play
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def get_mountaincar_env(render_mode):
    mc_env = gym.make('MountainCar-v0', render_mode=render_mode)
    # Note: Allows for faster P and R calculation,
    # computed policy works on default env params
    mc_env.env.env.env.force = 0.0015  # default: 0.001
    mc_env.env.env.env.max_speed = 0.105  # default: 0.07
    return mc_env

class DiscreteMountainCar:
    def __init__(self, position_bins=40, velocity_bins=40, gamma=0.9):
        # Create the environment
        self.env = get_mountaincar_env(render_mode=None)

        # Discretization parameters
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.gamma = gamma

        self.action_space = [0, 1, 2]

        # Create uniform discretization spaces
        self.position_space = np.linspace(
            self.env.observation_space.low[0],
            self.env.observation_space.high[0],
            position_bins
        )
        self.velocity_space = np.linspace(
            self.env.observation_space.low[1],
            self.env.observation_space.high[1],
            velocity_bins
        )

        self.n_actions = self.env.action_space.n

        # Initialize arrays

        self.transition_probs = np.zeros((
            position_bins,
            velocity_bins,
            self.n_actions,
            position_bins,
            velocity_bins
        ))

        self.transition_probs_v = np.zeros((
            position_bins,
            velocity_bins,
            self.n_actions,
            position_bins,
            velocity_bins
        ))

        self.rewards = np.zeros((position_bins, velocity_bins, self.n_actions))
        self.rewards_v = np.zeros((position_bins, velocity_bins, self.n_actions))

    def calculate_transition_probabilities_and_rewards(self, n_samples=10):
        """Calculate empirical transition probabilities."""
        counts = np.zeros_like(self.transition_probs)
        reward_sums = np.zeros_like(self.rewards)
        visit_counts = np.zeros_like(self.rewards)

        for i in range(n_samples):
            print('sample:', i)
            state, _ = self.env.reset()
            done = False

            while not done:
                pos_idx, vel_idx = self.discretize_state(state)

                # Try each action
                for action in range(self.n_actions):
                    # Save state for reset
                    orig_state = state.copy()

                    # Take action
                    next_state, reward, terminated, _truncated, _ = self.env.step(action)

                    next_pos_idx, next_vel_idx = self.discretize_state(next_state)

                    # Update counts and rewards
                    counts[pos_idx, vel_idx, action, next_pos_idx, next_vel_idx] += 1
                    reward_sums[pos_idx, vel_idx, action] += reward
                    visit_counts[pos_idx, vel_idx, action] += 1

                    # Reset state
                    self.env.env.state = orig_state

                # Progress simulation with random action
                action = self.env.action_space.sample()
                state, _reward, terminated, _truncated, _ = self.env.step(action)
                # NOTE: Don't want to truncate, otherwise the goal state is never found
                done = terminated

        # Calculate probabilities and average rewards
        state_action_counts = counts.sum(axis=(3, 4)) + 1e-10
        self.transition_probs = counts / state_action_counts[:, :, :, np.newaxis, np.newaxis]

        visit_counts = np.maximum(visit_counts, 1)
        self.rewards = reward_sums / visit_counts

        pickle_path = f'mountaincar-envP-{self.position_bins}x{self.velocity_bins}.pickle'
        with open(pickle_path, "wb") as f:
            pickle.dump([self.transition_probs, self.rewards], f)
        print(f'Transition + rewards matrices saved to {pickle_path}')

        return self.transition_probs, self.rewards

    def value_iteration(self, gamma, tol=1e-6, max_iterations=1000):
        """
        Value Iteration algorithm for Markov Decision Processes (MDPs) with a 2D grid state space.
        R and P are in their 2D form without flattening the state space.

        Notes:
        - R: Reward matrix of shape (rows, cols, A), where A is the number of actions.
        - P: Transition probability matrix of shape (rows, cols, A, rows, cols), where P[r, c, a, r', c'] is the probability
              of transitioning from state (r, c) to state (r', c') when action a is taken.
        - grid_shape: Tuple (rows, cols) representing the shape of the 2D grid.
        - gamma: Discount factor (default is 0.9).
        - theta: Convergence threshold (default is 1e-6).

        Returns:
        - V: Value function (state values) reshaped to a 2D grid.
        - policy: Optimal policy (best action for each state) reshaped to a 2D grid.
        """

        V = np.zeros((self.position_bins, self.velocity_bins))  # value function
        P = self.transition_probs
        R = self.rewards

        for iteration in range(max_iterations):
            # Q_values: Shape (rows, cols, A): the expected rewards for each (state, action) combination
            Q_values = np.sum(P * V[np.newaxis, np.newaxis, np.newaxis, :, :],
                              axis=(3, 4))  # Sum over next states (r', c') for each action

            # Add rewards and discount factor to Q_values
            Q_values = R + gamma * Q_values  # Shape (rows, cols, A)

            # For each state (r, c), find the action with the maximum Q-value
            V_new = np.max(Q_values, axis=2)  # (Shape: (rows, cols))

            delta = np.max(np.abs(V_new - V))

            # Update the value function
            V = V_new

            # Check convergence
            if delta < tol and iteration > 100:  # Ensure minimum iterations
                print(f"Value iteration converged after {iteration + 1} iterations")
                break

        policy = np.argmax(Q_values, axis=2)  # Shape: (rows, cols). Fetch policy from converged value function matrix

        return V, policy

    def policy_iteration(self, gamma, tol=1e-6, max_iterations=200, max_eval_iterations=100):
        """
        Policy iteration algorithm that alternates between policy evaluation and improvement.

        Args:
            max_iterations (int): Maximum number of policy iteration loops
            max_eval_iterations (int): Maximum iterations for policy evaluation
            gather_data (bool): If True, collects data on policy changes at each iteration

        Returns:
            tuple: (value_function, policy)
        """
        # Initialize random policy and value function

        def _policy_evaluation(policy, value_function, gamma, tol, max_iterations):
            """
            Evaluate a policy by computing its value function.

            Args:
                policy (np.ndarray): Current policy
                value_function (np.ndarray): Initial value function estimate
                tol (float): Convergence threshold
                max_iterations (int): Maximum number of iterations

            Returns:
                np.ndarray: Updated value function
            """
            for iteration in range(max_iterations):
                old_value = value_function.copy()
                # For each state
                for pos in range(self.position_bins):
                    for vel in range(self.velocity_bins):
                        action = policy[pos, vel]

                        # Calculate expected value
                        future_value = np.sum(self.transition_probs[pos, vel, action] * value_function)
                        value_function[pos, vel] = (self.rewards[pos, vel, action] + gamma * future_value)

                # Check for convergence
                delta = np.max(np.abs(value_function - old_value))
                if delta < tol:
                    break

            return value_function

        def _policy_improvement(value_function, gamma):
            """
            Improve policy based on current value function.

            Args:
                value_function (np.ndarray): Current value function

            Returns:
                np.ndarray: Improved policy
            """
            # Calculate action values for all states
            future_values = np.sum(
                self.transition_probs * value_function[np.newaxis, np.newaxis, np.newaxis, :, :],
                axis=(3, 4)
            )
            action_values = self.rewards + gamma * future_values

            # Select best action for each state
            return np.argmax(action_values, axis=2)

        V = np.zeros((self.position_bins, self.velocity_bins))  # Value function (state values)
        policy = np.random.randint(0, self.n_actions,
                                   size=(self.position_bins, self.velocity_bins))  # Initial random policy

        for iteration in range(max_iterations):
            old_policy = policy.copy()

            # Policy Evaluation
            V = _policy_evaluation(policy, V, gamma, tol, max_eval_iterations)

            # Policy Improvement
            policy = _policy_improvement(V, gamma)

            # Check for policy convergence
            if np.array_equal(old_policy, policy):
                print(f"Policy iteration converged after {iteration + 1} iterations")
                break

        return V, policy

    def q_learn(self, gamma, episodes, verbose=False):
        """
        Implements Q-learning for the Mountain Car environment with convergence check.
        Returns the trained Q-table and training statistics.
        """
        # Initialize Q-table as a 3D numpy array
        Q = np.zeros((self.position_bins + 1, self.velocity_bins + 1, self.n_actions))

        # Hyperparameters
        convergence_threshold = 1e-7
        convergence_window = 100  # Check convergence after at least 100 episodes
        q_value_deltas = []

        # Pre-allocate arrays for storing experience
        episode_rewards = np.zeros(episodes)
        episode_lengths = np.zeros(episodes)
        mse_list = []

        # Training loop
        for episode in range(episodes):
            state = self.env.reset()[0]
            rng = np.random.default_rng(seed=42+episode)
            pos_bin, vel_bin = self.discretize_state(state)
            total_reward = 0
            steps = 0
            done = False

            # Store Q-values before episode for convergence check
            Q_prev = Q.copy()

            while not done:
                # Epsilon-greedy action selection
                epsilon = 1 / np.sqrt(0.02 * episode + 2)  # Inverse sqrt decay
                if rng.random() < epsilon:
                    action = rng.choice(self.action_space)
                else:
                    action = np.argmax(Q[pos_bin, vel_bin])

                # Softmax action selection
                # action = self.softmax(Q, pos_bin, vel_bin, rng, tau=epsilon)

                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_pos_bin, next_vel_bin = self.discretize_state(next_state)

                # Vectorized Q-learning update
                best_next_value = np.max(Q[next_pos_bin, next_vel_bin])
                td_target = reward + gamma * best_next_value
                td_error = td_target - Q[pos_bin, vel_bin, action]

                # Decaying learning rate/epsilon/tau update
                alpha = 1 / np.sqrt(0.02 * episode + 2)  # Inverse sqrt decay
                Q[pos_bin, vel_bin, action] += alpha * td_error

                # Update state
                pos_bin, vel_bin = next_pos_bin, next_vel_bin
                total_reward += reward
                steps += 1

            # Update statistics using vectorized operations
            episode_rewards[episode] = total_reward
            episode_lengths[episode] = steps

            # Track mean Q-value deltas
            delta = np.mean(np.abs(Q - Q_prev))
            q_value_deltas.append(delta)

            # Convergence check: use mean squared error of Q-delta
            if len(q_value_deltas) >= 100:
                recent_deltas = np.array(q_value_deltas[-100:])
                mse = np.square(recent_deltas - np.mean(recent_deltas)).mean(axis=0)
                mse_list.append(mse)
                if mse < convergence_threshold:
                    print(f"Converged after {episode} episodes")
                    break

            if delta < 1e-6:
                print(f"Converged after {episode} episodes")
                break


            # Report stats every N episodes
            if (episode + 1) % convergence_window == 0:
                # Print progress and delta
                if verbose:
                    recent_rewards = episode_rewards[max(0, episode - 99):episode + 1]
                    recent_lengths = episode_lengths[max(0, episode - 99):episode + 1]
                    print(f"{episode + 1},{np.mean(recent_rewards):.2f},{np.max(recent_rewards):.2f},"
                          f"{np.mean(recent_lengths):.2f},{epsilon:.3f},{delta:.6f}")

        return Q, episode_rewards, episode_lengths

    def execute_policy(self, policy, n_episodes, render=False):
        """Execute policy."""
        if render:
            self.env = get_mountaincar_env(render_mode='human')

        episode_rewards = []
        episode_lengths = []
        success_count = 0
        max_x_position_reached = -np.inf

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done:  # default: truncates at 200 timesteps
                pos_idx, vel_idx = self.discretize_state(state)
                action = policy[pos_idx, vel_idx]

                state, reward, terminated, truncated, _info = self.env.step(action)
                done = terminated or truncated

                max_x_position_reached = max(max_x_position_reached, state[0])
                episode_reward += reward
                steps += 1

                if state[0] >= 0.5:
                    success_count += 1
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

        stats = {
            'mean_reward': np.mean(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'success_rate': success_count / n_episodes,
            'max_x_position_reached': max_x_position_reached,
            'episode_rewards': episode_rewards,
        }

        return stats

    def discretize_state(self, state):
        """Convert continuous state to discrete state indices."""
        position, velocity = state

        position_idx = np.digitize(position, self.position_space) - 1
        velocity_idx = np.digitize(velocity, self.velocity_space) - 1

        position_idx = np.clip(position_idx, 0, self.position_bins - 1)
        velocity_idx = np.clip(velocity_idx, 0, self.velocity_bins - 1)

        return position_idx, velocity_idx

    def print_policy_performance(self, stats):
        """Print detailed performance statistics."""
        print("\nPolicy Performance Statistics:")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Max Reward: {stats['max_reward']:.2f}")
        print(f"Success Rate: {stats['success_rate'] * 100:.2f}%")

def plot_policy_matrix(matrix, dest_path=None):
    """
    Plot a 2D policy matrix with discrete values.

    Parameters:
    matrix: 2D numpy array containing values 0, 1, 2
    cmap: colormap to use (default: 'viridis')
    """
    DPI = 240
    FONTSIZE = 14
    FIGSIZE = (5, 4)

    # Create figure and axis
    plt.figure(figsize=FIGSIZE)

    # Create the plot
    plt.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')

    plt.colorbar(label='Policy Value', ticks=[0, 1, 2])
    plt.xlabel('Position Bin', fontsize=FONTSIZE)
    plt.ylabel('Velocity Bin', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.gca().invert_yaxis()  # Invert y-axis to match example
    plt.grid(False)

    if dest_path:
        plt.savefig(dest_path, bbox_inches='tight', dpi=DPI)
    else:
        plt.show(block=False)
    plt.close()

def execute_vi(env: DiscreteMountainCar, gamma):
    vi_start = time.time()
    vi_value_function, vi_policy = env.value_iteration(gamma, tol=1e-6, max_iterations=1000)
    vi_time = time.time() - vi_start
    p_bins, v_bins = env.position_bins, env.velocity_bins
    plot_policy_matrix(vi_policy, dest_path=None)
    print('Executing VI policy...')
    vi_stats = env.execute_policy(vi_policy, n_episodes=100, render=False)
    print('time,success_rate,mean_reward,max_reward')
    print(f'{vi_time:.2f},{vi_stats["success_rate"]:.2f},{vi_stats["mean_reward"]},{vi_stats["max_reward"]}')
    return vi_time, vi_stats['mean_reward']


def execute_pi(env: DiscreteMountainCar, gamma):
    pi_start = time.time()
    pi_value_function, pi_policy = env.policy_iteration(gamma, tol=1e-6)
    pi_time = time.time() - pi_start
    p_bins, v_bins = env.position_bins, env.velocity_bins
    plot_policy_matrix(pi_policy, dest_path=None)

    print('Executing PI policy...')
    pi_stats = env.execute_policy(pi_policy, n_episodes=100, render=False)
    print('time,success_rate,mean_reward,max_reward')
    print(f'{pi_time:.2f},{pi_stats["success_rate"]:.2f},{pi_stats["mean_reward"]},{pi_stats["max_reward"]}')
    return pi_time, pi_stats['mean_reward']


def execute_ql(env: DiscreteMountainCar, gamma):
    ql_start = time.time()
    q_table, _, _ = env.q_learn(gamma, episodes=10000, verbose=False)
    ql_time = time.time() - ql_start
    ql_policy = np.argmax(q_table, axis=2)  # fetch policy based on actions with highest reward
    p_bins, v_bins = env.position_bins, env.velocity_bins
    plot_policy_matrix(ql_policy, dest_path=None)

    print('Executing Q-learning policy...')
    ql_stats = env.execute_policy(ql_policy, n_episodes=100)
    print('time,success_rate,mean_reward,max_reward')
    print(f'{ql_time:.2f},{ql_stats["success_rate"]:.2f},{ql_stats["mean_reward"]},{ql_stats["max_reward"]}')
    return ql_time, ql_stats['mean_reward']


def run_mountain_car(mountain_car_env, generate_P=False, path_P=None):
    if generate_P:
        print("Calculating transition probabilities...")
        mountain_car_env.calculate_transition_probabilities_and_rewards()
    else:
        probs_rewards_path = path_P
        with open(probs_rewards_path, "rb") as f:
            mountain_car_env.transition_probs, mountain_car_env.rewards = pickle.load(f)
        print("Successfully fetched transition probabilities and rewards matrices.")

    print("\nPerforming value iteration...")
    execute_vi(mountain_car_env, gamma=0.9)

    print("\nPerforming policy iteration...")
    execute_pi(mountain_car_env, gamma=0.9)

    print("\nPerforming Q-learning...")
    execute_ql(mountain_car_env, gamma=0.9)


if __name__ == '__main__':
    env = DiscreteMountainCar(40, 40)
    run_mountain_car(env, path_P='mountaincar-envP-40x40.pickle')