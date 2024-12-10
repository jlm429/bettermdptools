import unittest
import warnings

import gymnasium as gym
import numpy as np

from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress warnings during test setup
        warnings.filterwarnings("ignore")

        cls.frozen_lake = gym.make("FrozenLake8x8-v1", render_mode=None)

    def test_value_iteration_heatmap(self):
        V, V_track, pi = Planner(self.frozen_lake.P).value_iteration(n_iters=100)
        size = (8, 8)

        # Check if the values heat map function runs without errors
        try:
            Plots.values_heat_map(
                V, "Frozen Lake\nValue Iteration State Values", size, show=False
            )
        except Exception as e:
            self.fail(f"values_heat_map raised an exception: {e}")

    def test_value_iteration_v_iters_plot(self):
        V, V_track, pi = Planner(self.frozen_lake.P).value_iteration(n_iters=100)

        # Clip trailing zeros in case convergence is reached before max iterations
        max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), "b")

        # Check if the v_iters_plot function runs without errors
        try:
            Plots.v_iters_plot(
                max_value_per_iter,
                "Frozen Lake\nMean Value v Iterations",
                show=False,
            )
        except Exception as e:
            self.fail(f"v_iters_plot raised an exception: {e}")

    def test_policy_map_plot(self):
        V, V_track, pi = Planner(self.frozen_lake.P).value_iteration(n_iters=100)

        fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        fl_map_size = (8, 8)
        title = "FL Mapped Policy\nArrows represent best action"
        val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)

        # Check if the v_iters_plot function runs without errors
        try:
            Plots.plot_policy(val_max, policy_map, fl_map_size, title, show=False)
        except Exception as e:
            self.fail(f"v_iters_plot raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
