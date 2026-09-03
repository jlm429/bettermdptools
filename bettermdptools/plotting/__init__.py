"""Composable plotting for BetterMDPTools.

Preparation functions are pure and independent of Matplotlib. Renderers draw
only on explicit caller-provided :class:`matplotlib.axes.Axes` objects and
return those axes. They never show, save, close, resize, or apply figure-level
layout. Callers own the figure and its lifecycle.
"""

import matplotlib.axes

from ._matplotlib import (
    ConvergenceAxes,
    ValuePolicyAxes,
    plot_convergence,
    plot_learning_curve,
    plot_policy_convergence,
    plot_policy_grid,
    plot_value_convergence,
    plot_value_heatmap,
    plot_value_policy,
)
from ._prepare import (
    ConvergenceData,
    LearningCurveData,
    PolicyGridData,
    ValueGridData,
    aggregate_values,
    prepare_convergence,
    prepare_learning_curve,
    prepare_policy_grid,
    prepare_value_grid,
)

__all__ = [
    "ConvergenceAxes",
    "ConvergenceData",
    "LearningCurveData",
    "PolicyGridData",
    "ValueGridData",
    "ValuePolicyAxes",
    "aggregate_values",
    "plot_convergence",
    "plot_learning_curve",
    "plot_policy_convergence",
    "plot_policy_grid",
    "plot_value_convergence",
    "plot_value_heatmap",
    "plot_value_policy",
    "prepare_convergence",
    "prepare_learning_curve",
    "prepare_policy_grid",
    "prepare_value_grid",
]
