import unittest
from io import BytesIO
from unittest.mock import patch

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from bettermdptools.plotting import (
    aggregate_values,
    plot_convergence,
    plot_learning_curve,
    plot_policy_grid,
    plot_value_heatmap,
    plot_value_policy,
    prepare_convergence,
    prepare_learning_curve,
    prepare_policy_grid,
    prepare_value_grid,
)


class TestPlotPreparation(unittest.TestCase):
    def test_value_grid_preserves_numeric_values_and_formats_annotations(self):
        source = np.array([1.234, np.nan])

        data = prepare_value_grid(source, (1, 2), decimals=2)

        self.assertAlmostEqual(data.values[0, 0], 1.234)
        self.assertTrue(np.isnan(data.values[0, 1]))
        self.assertEqual(data.annotations.tolist(), [["1.23", ""]])
        self.assertIsNot(data.values.base, source)

    def test_policy_grid_preserves_labels_and_validates_coverage(self):
        data = prepare_policy_grid(
            {0: 0, 1: 1},
            [1.0, 2.0],
            {0: "MOVE LEFT", 1: "MOVE RIGHT"},
            (1, 2),
        )

        self.assertEqual(data.actions.tolist(), [["MOVE LEFT", "MOVE RIGHT"]])
        with self.assertRaisesRegex(ValueError, "missing state 1"):
            prepare_policy_grid({0: 0}, [1.0, 2.0], None, (1, 2))
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            prepare_policy_grid({0: 0.5}, [1.0], None, (1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            prepare_policy_grid({0: -1}, [1.0], None, (1, 1))

    def test_aggregate_values_uses_simultaneous_original_axes(self):
        values = np.arange(24.0).reshape(2, 3, 4)
        expected = np.mean(values, axis=(0, 2))

        for axes in ((0, 2), (2, 0), (-3, -1)):
            with self.subTest(axes=axes):
                np.testing.assert_array_equal(
                    aggregate_values(values, values.shape, axes), expected
                )
        unchanged = aggregate_values(values, values.shape, ())
        np.testing.assert_array_equal(unchanged, values)
        self.assertIsNot(unchanged, values)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            aggregate_values(values, values.shape, (0, -3))

    def test_learning_curve_summarizes_runs_after_per_run_smoothing(self):
        data = prepare_learning_curve([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]], window=2)

        np.testing.assert_array_equal(data.episodes, [1, 2, 3])
        np.testing.assert_allclose(data.smoothed, [[0.0, 1.0, 3.0], [2.0, 3.0, 5.0]])
        np.testing.assert_allclose(data.center, [1.0, 2.0, 4.0])
        np.testing.assert_allclose(data.lower, [0.2, 1.2, 3.2])
        np.testing.assert_allclose(data.upper, [1.8, 2.8, 4.8])

        single = prepare_learning_curve([0.0, 1.0])
        self.assertEqual(single.rewards.shape, (1, 2))
        self.assertIsNone(single.lower)
        self.assertIsNone(single.upper)

        with self.assertRaisesRegex(ValueError, "one run and one episode"):
            prepare_learning_curve(np.empty((0, 2)))

    def test_convergence_uses_only_explicit_validity_and_keeps_zero_rows(self):
        values = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, -2.0], [0.0, 0.0]])
        policies = np.array([[0, 0], [0, 1], [1, 1], [0, 0]])

        limited = prepare_convergence(values, policy_history=policies, valid_length=3)
        complete = prepare_convergence(values)

        np.testing.assert_array_equal(limited.iterations, [1, 2])
        np.testing.assert_array_equal(limited.value_delta, [0.0, 2.0])
        np.testing.assert_array_equal(limited.policy_changes, [1, 1])
        np.testing.assert_array_equal(complete.value_delta, [0.0, 2.0, 2.0])

        q_history = np.arange(24.0).reshape(3, 2, 4)
        q_convergence = prepare_convergence(q_history, value_statistic="mean")
        np.testing.assert_array_equal(q_convergence.value_delta, [8.0, 8.0])

        with self.assertRaisesRegex(ValueError, "must not contain NaN"):
            prepare_convergence([[0.0], [np.nan]])


class TestAxesRenderers(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_renderers_use_exact_axes_without_show_save_close_or_global_state(self):
        target_figure, axes = plt.subplots(2, 2)
        value_ax, policy_ax, learning_ax, value_convergence_ax = axes.flat
        policy_convergence_figure, policy_convergence_ax = plt.subplots()
        current_figure, current_ax = plt.subplots()
        plt.sca(current_ax)
        figures_before = plt.get_fignums()
        rc_before = {name: repr(value) for name, value in mpl.rcParams.items()}
        policy = prepare_policy_grid(
            {0: 0, 1: 1}, [1.0, 2.0], {0: "LEFT", 1: "RIGHT"}, (1, 2)
        )
        curve = prepare_learning_curve([[0.0, 1.0], [1.0, 2.0]])
        convergence = prepare_convergence(
            [[0.0, 0.0], [1.0, 0.0]], policy_history=[[0, 0], [1, 0]]
        )

        with (
            patch("matplotlib.pyplot.show") as show,
            patch("matplotlib.pyplot.savefig") as save,
            patch("matplotlib.pyplot.close") as close,
            patch.object(target_figure, "tight_layout") as tight_layout,
            patch.object(target_figure, "set_layout_engine") as set_layout_engine,
            patch.object(target_figure, "set_size_inches") as set_size_inches,
        ):
            self.assertIs(
                plot_value_heatmap(prepare_value_grid([1.0, 2.0], (1, 2)), ax=value_ax),
                value_ax,
            )
            self.assertIs(plot_policy_grid(policy, ax=policy_ax), policy_ax)
            self.assertIs(plot_learning_curve(curve, ax=learning_ax), learning_ax)
            convergence_axes = plot_convergence(
                convergence,
                value_ax=value_convergence_ax,
                policy_ax=policy_convergence_ax,
            )

        show.assert_not_called()
        save.assert_not_called()
        close.assert_not_called()
        tight_layout.assert_not_called()
        set_layout_engine.assert_not_called()
        set_size_inches.assert_not_called()
        self.assertIs(convergence_axes.values, value_convergence_ax)
        self.assertIs(convergence_axes.policy, policy_convergence_ax)
        self.assertIs(plt.gcf(), current_figure)
        self.assertIs(plt.gca(), current_ax)
        self.assertEqual(plt.get_fignums(), figures_before)
        self.assertEqual(
            {name: repr(value) for name, value in mpl.rcParams.items()}, rc_before
        )
        self.assertEqual(len(current_ax.collections), 0)
        self.assertEqual(len(current_ax.lines), 0)
        self.assertEqual(len(value_ax.collections), 1)
        self.assertEqual(len(policy_ax.collections), 1)
        self.assertEqual(len(learning_ax.lines), 3)
        self.assertEqual(len(learning_ax.collections), 1)
        self.assertTrue(
            all(line.get_visible() for line in learning_ax.get_xgridlines())
        )
        self.assertTrue(
            all(line.get_visible() for line in learning_ax.get_ygridlines())
        )
        self.assertEqual(
            [text.get_text() for text in learning_ax.get_legend().get_texts()],
            ["mean", "10% to 90% across runs"],
        )
        self.assertEqual(len(value_convergence_ax.lines), 1)
        self.assertEqual(len(policy_convergence_ax.lines), 1)
        self.assertTrue(plt.fignum_exists(target_figure.canvas.manager.num))
        self.assertTrue(plt.fignum_exists(policy_convergence_figure.canvas.manager.num))

    def test_colorbar_target_and_value_policy_composition(self):
        figure = plt.figure()
        grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 0.05))
        value_ax = figure.add_subplot(grid[0, 0])
        policy_ax = figure.add_subplot(grid[0, 1])
        colorbar_ax = figure.add_subplot(grid[0, 2])
        data = prepare_policy_grid(
            {0: 0, 1: 1}, [0.0, 1.0], {0: "LEFT", 1: "RIGHT"}, (1, 2)
        )

        axes = plot_value_policy(
            data,
            value_ax=value_ax,
            policy_ax=policy_ax,
            cbar_ax=colorbar_ax,
        )

        self.assertIs(axes.values, value_ax)
        self.assertIs(axes.policy, policy_ax)
        self.assertEqual(figure.axes, [value_ax, policy_ax, colorbar_ax])
        self.assertEqual(colorbar_ax.get_ylabel(), "State value")
        self.assertEqual(
            [text.get_text() for text in policy_ax.texts], ["LEFT", "RIGHT"]
        )

    def test_save_through_axes_figure_smoke(self):
        figure, ax = plt.subplots()
        plot_learning_curve(prepare_learning_curve([0.0, 1.0, 2.0]), ax=ax)
        destination = BytesIO()

        ax.figure.savefig(destination, format="png")

        self.assertGreater(len(destination.getvalue()), 100)


if __name__ == "__main__":
    unittest.main()
