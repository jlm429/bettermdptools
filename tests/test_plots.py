import unittest
import warnings
from unittest.mock import patch

import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress warnings during test setup
        warnings.filterwarnings("ignore")

        cls.frozen_lake = gym.make("FrozenLake8x8-v1", render_mode=None)

    @classmethod
    def tearDownClass(cls):
        cls.frozen_lake.close()

    def test_value_iteration_heatmap(self):
        V, V_track, pi = Planner(self.frozen_lake.unwrapped.P).value_iteration(
            n_iters=100
        )
        size = (8, 8)

        # Check if the values heat map function runs without errors
        try:
            Plots.values_heat_map(
                V, "Frozen Lake\nValue Iteration State Values", size, show=False
            )
        except Exception as e:
            self.fail(f"values_heat_map raised an exception: {e}")

    def test_value_iteration_v_iters_plot(self):
        V, V_track, pi = Planner(self.frozen_lake.unwrapped.P).value_iteration(
            n_iters=100
        )

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
        V, V_track, pi = Planner(self.frozen_lake.unwrapped.P).value_iteration(
            n_iters=100
        )

        fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        fl_map_size = (8, 8)
        title = "FL Mapped Policy\nArrows represent best action"
        val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)

        # Check if the v_iters_plot function runs without errors
        try:
            Plots.plot_policy(val_max, policy_map, fl_map_size, title, show=False)
        except Exception as e:
            self.fail(f"v_iters_plot raised an exception: {e}")

    def test_policy_map_preserves_multi_character_action_labels(self):
        values = np.array([1.0, 2.0])
        policy = {0: 0, 1: 1}

        _, policy_map = Plots.get_policy_map(
            policy,
            values,
            {0: "LEFT", 1: "RIGHT"},
            (1, 2),
        )

        self.assertEqual(policy_map.tolist(), [["LEFT", "RIGHT"]])

    def test_policy_plot_fits_multi_character_labels_within_cells(self):
        values = np.arange(16.0).reshape(4, 4)
        labels = np.array(
            [["MOVE LEFT", "MOVE RIGHT", "STAY PUT", "TURN AROUND"]] * 4,
            dtype=object,
        )
        figure, ax = plt.subplots()

        try:
            Plots.plot_policy(values, labels, (4, 4), "Policy", show=False, ax=ax)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            oversized = []

            for position, annotation in enumerate(ax.texts):
                row, column = divmod(position, values.shape[1])
                annotation_bounds = annotation.get_window_extent(renderer)
                corner_a = ax.transData.transform((column, row))
                corner_b = ax.transData.transform((column + 1, row + 1))
                cell_width = abs(corner_b[0] - corner_a[0])
                cell_height = abs(corner_b[1] - corner_a[1])

                if (
                    annotation_bounds.width > cell_width
                    or annotation_bounds.height > cell_height
                ):
                    oversized.append((annotation.get_text(), position))

            self.assertEqual(oversized, [])
        finally:
            plt.close(figure)

    def test_policy_aggregation_averages_values_not_categorical_labels(self):
        values = np.arange(8.0)
        policy = {state: state % 2 for state in range(8)}

        aggregated = Plots.get_values_agg_axis_means(
            policy,
            values,
            (2, 2, 2),
            (0,),
        )

        np.testing.assert_array_equal(aggregated, [[2.0, 3.0], [4.0, 5.0]])

    def test_aggregation_axes_use_original_shape_with_order_independent_semantics(
        self,
    ):
        values = np.arange(24.0).reshape(2, 3, 4)
        expected = np.mean(values, axis=(0, 2))

        for axes in ((0, 2), (2, 0), (-3, -1)):
            with self.subTest(axes=axes):
                np.testing.assert_array_equal(
                    Plots.aggregate_values(values, values.shape, axes),
                    expected,
                )

        unchanged = Plots.aggregate_values(values, values.shape, ())
        np.testing.assert_array_equal(unchanged, values)
        self.assertIsNot(unchanged, values)

        with self.assertRaisesRegex(ValueError, "duplicate axes"):
            Plots.aggregate_values(values, values.shape, (0, -3))
        with self.assertRaisesRegex(TypeError, "numeric measurement"):
            Plots.aggregate_values(
                np.array([["LEFT", "RIGHT"]], dtype=object),
                (1, 2),
                (0,),
            )

    def test_plot_data_transformations_are_pure_and_preserve_full_labels(self):
        values = np.array([1.234, 5.678])
        original = values.copy()

        frame = Plots.values_to_dataframe(values, (1, 2))
        iterations = Plots.iterations_to_dataframe([[1.0, 2.0], [3.0, 4.0]])
        _, labels = Plots.get_policy_map(
            {0: 0, 1: 1},
            values,
            {0: "MOVE LEFT", 1: "MOVE RIGHT"},
            (1, 2),
        )
        label_frame = pd.DataFrame(labels)

        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(frame.to_numpy(), [[1.23, 5.68]])
        np.testing.assert_array_equal(iterations.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(label_frame.to_numpy().tolist(), [["MOVE LEFT", "MOVE RIGHT"]])

    def test_renderers_use_supplied_axes_without_global_style_or_axes_leaks(self):
        rc_before = {key: repr(value) for key, value in mpl.rcParams.items()}
        figure, axes = plt.subplots(2, 2)
        value_ax, iteration_ax, policy_ax, current_ax = axes.flat
        plt.sca(current_ax)
        values = np.array([1.0, 2.0])
        mapped_values, labels = Plots.get_policy_map(
            {0: 0, 1: 1},
            values,
            {0: "MOVE LEFT", 1: "MOVE RIGHT"},
            (1, 2),
        )

        try:
            self.assertIs(
                Plots.values_heat_map(
                    values,
                    "Values",
                    (1, 2),
                    show=False,
                    ax=value_ax,
                ),
                value_ax,
            )
            self.assertIs(
                Plots.v_iters_plot(
                    [1.0, 2.0],
                    "Iterations",
                    show=False,
                    ax=iteration_ax,
                ),
                iteration_ax,
            )
            self.assertIs(
                Plots.plot_policy(
                    mapped_values,
                    labels,
                    (1, 2),
                    "Policy",
                    show=False,
                    ax=policy_ax,
                ),
                policy_ax,
            )

            self.assertIs(plt.gca(), current_ax)
            self.assertEqual(current_ax.get_title(), "")
            self.assertEqual(len(current_ax.collections), 0)
            self.assertEqual(
                [text.get_text() for text in policy_ax.texts],
                ["MOVE LEFT", "MOVE RIGHT"],
            )
            self.assertEqual(
                {key: repr(value) for key, value in mpl.rcParams.items()},
                rc_before,
            )
        finally:
            plt.close(figure)

    def test_renderer_fallback_shows_without_changing_pyplot_target(self):
        caller_figure, caller_ax = plt.subplots()
        plt.sca(caller_ax)
        fallback_ax = None
        fallback_number = None

        try:
            with patch("bettermdptools.utils.plots.plt.show") as show:
                fallback_ax = Plots.values_heat_map(
                    np.array([1.0, 2.0]),
                    "Values",
                    (1, 2),
                )

            show.assert_called_once_with()
            self.assertIsNot(fallback_ax, caller_ax)
            self.assertIs(plt.gcf(), caller_figure)
            self.assertIs(plt.gca(), caller_ax)
            fallback_manager = fallback_ax.figure.canvas.manager
            fallback_number = fallback_manager.num
            self.assertTrue(plt.fignum_exists(fallback_number))

            line = plt.plot([0, 1], [1, 0])[0]

            self.assertIs(line.axes, caller_ax)
            self.assertEqual(len(fallback_ax.lines), 0)

            with patch.object(
                fallback_manager,
                "destroy",
                wraps=fallback_manager.destroy,
            ) as destroy:
                plt.close(fallback_ax.figure)

            destroy.assert_called_once_with()
            self.assertFalse(plt.fignum_exists(fallback_number))
            self.assertIs(plt.gca(), caller_ax)
        finally:
            if fallback_number is not None and plt.fignum_exists(fallback_number):
                plt.close(fallback_number)
            plt.close(caller_figure)

    def test_renderer_fallback_without_prior_figure_remains_current(self):
        plt.close("all")
        fallback_ax = None
        fallback_number = None

        try:
            with patch("bettermdptools.utils.plots.plt.show") as show:
                fallback_ax = Plots.values_heat_map(
                    np.array([1.0, 2.0]),
                    "Values",
                    (1, 2),
                )

            show.assert_called_once_with()
            fallback_manager = fallback_ax.figure.canvas.manager
            fallback_number = fallback_manager.num
            self.assertTrue(plt.fignum_exists(fallback_number))
            self.assertEqual(plt.get_fignums(), [fallback_number])
            self.assertIs(plt.gca(), fallback_ax)

            line = plt.plot([0, 1], [1, 0])[0]

            self.assertIs(line.axes, fallback_ax)

            with patch.object(
                fallback_manager,
                "destroy",
                wraps=fallback_manager.destroy,
            ) as destroy:
                plt.close(fallback_ax.figure)

            destroy.assert_called_once_with()
            self.assertFalse(plt.fignum_exists(fallback_number))
        finally:
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
