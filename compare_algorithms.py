"""Enable changes to be compared to previous versions of the code."""

import argparse
import time

import gymnasium as gym
import numpy as np

from bettermdptools.algorithms.planner import Planner
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper

# from bettermdptools.envs.pendulum_wrapper import PendulumWrapper
# from bettermdptools.envs.acrobot_wrapper import AcrobotWrapper
from bettermdptools.seed import set_seed


def run_value_iteration(env, n_iters=1000, vectorized=False, dtype=np.float64):
    """
    Run value iteration on the given environment.

    Parameters:
    env (gym.Env): The environment to run value iteration on.
    n_iters (int): Number of iterations to run.
    vectorized (bool): Whether to use the vectorized version of value iteration.
    dtype (np.dtype): Data type to use for computations.

    Returns:
    tuple: Value function, value function track, and policy.
    """
    planner = Planner(env.P)
    if vectorized:
        V, V_track, pi = planner.value_iteration_vectorized(
            n_iters=n_iters, dtype=dtype
        )
    else:
        V, V_track, pi = planner.value_iteration(n_iters=n_iters, dtype=dtype)
    return V, V_track, pi


def compare_outputs(V1, V_track1, pi1, V2, V_track2, pi2, atol=1e-8, rtol=1e-5):
    """
    Compare the outputs of two value iteration runs.

    Parameters:
    V1, V2 (np.ndarray): Value functions to compare.
    V_track1, V_track2 (np.ndarray): Value function tracks to compare.
    pi1, pi2 (dict): Policies to compare.
    atol (float): Absolute tolerance for comparison.
    rtol (float): Relative tolerance for comparison.

    Raises:
    AssertionError: If any of the comparisons fail.
    """
    assert V1.shape == V2.shape, "Value arrays are not equal"
    assert V_track1.shape == V_track2.shape, "Value track arrays are not equal"
    assert all(
        pi1[k].shape == pi2[k].shape for k in pi1.keys()
    ), "Policy values are not equal"
    print("All outputs are equivalently shaped")

    assert np.allclose(V1, V2, atol=atol, rtol=rtol), "Value arrays are not equal"
    # assert np.allclose(V_track1, V_track2, atol=atol, rtol=rtol), "Value track arrays are not equal"
    assert set(pi1.keys()) == set(pi2.keys()), "Policy keys are not equal"
    assert all(
        np.array_equal(pi1[k], pi2[k]) for k in pi1.keys()
    ), "Policy values are not equal"
    print("All outputs are equal")


def main():
    """
    Main function to run and compare value iteration on specified environments.
    """
    parser = argparse.ArgumentParser(
        description="Compare value iteration algorithms on various environments."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        help="Environment to test: blackjack, cartpole, frozenlake, taxi, or all",
    )
    parser.add_argument(
        "--n_iters", type=int, default=1000, help="Number of iterations to run"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Data type to use: float64 or float32",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    environments = {
        "blackjack": (BlackjackWrapper, "Blackjack-v1"),
        "cartpole": (CartpoleWrapper, "CartPole-v1"),
        "frozenlake": (lambda x: x, "FrozenLake8x8-v1"),
        "taxi": (lambda x: x, "Taxi-v3"),
        # TODO: these 2 are not functional right now
        # "pendulum": (PendulumWrapper, "Pendulum-v0"),
        # "acrobot": (AcrobotWrapper, "Acrobot-v1"),
    }

    if args.env.lower() == "all":
        env_list = environments.keys()
    else:
        env_list = [args.env.lower()]

    if args.dtype == "float64":
        dtype = np.float64
    elif args.dtype == "float32":
        dtype = np.float32
    else:
        raise ValueError("Invalid dtype. Use 'float64' or 'float32'")

    for env_name in env_list:
        print(f"\n{'='*20} Testing {env_name} {'='*20}")
        wrapper_class, gym_env_name = environments[env_name]
        env = wrapper_class(gym.make(gym_env_name, render_mode=None))

        # Run and time value iteration (original)
        print("Running value iteration (original)...")
        start_time = time.time()
        V1, V_track1, pi1 = run_value_iteration(
            env, n_iters=args.n_iters, vectorized=False, dtype=dtype
        )
        original_time = time.time() - start_time
        print(f"Original value iteration time: {original_time:.4f} seconds")

        # Run and time value iteration (vectorized)
        print("Running value iteration (vectorized)...")
        start_time = time.time()
        V2, V_track2, pi2 = run_value_iteration(
            env, n_iters=args.n_iters, vectorized=True, dtype=dtype
        )
        vectorized_time = time.time() - start_time
        print(f"Vectorized value iteration time: {vectorized_time:.4f} seconds")

        # Compare outputs
        print("Comparing value iteration outputs...")
        try:
            compare_outputs(V1, V_track1, pi1, V2, V_track2, pi2)
        except AssertionError as e:
            print(f"Comparison failed for {env_name}: {str(e)}")


if __name__ == "__main__":
    main()
