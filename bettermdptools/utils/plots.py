# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.exceptions import AxisError


class Plots:
    """Transform and visualize values, policies, and training traces.

    Renderers restore an existing pyplot figure and axes after creating a
    fallback. If no pyplot figure existed, the managed fallback remains current
    so pyplot show and close behavior stays intact.
    """

    @staticmethod
    def values_to_dataframe(
        data: Any, size: Sequence[int], *, decimals: int = 2
    ) -> pd.DataFrame:
        """Return reshaped, rounded values without changing the input."""
        values = np.asarray(data).reshape(tuple(size))
        return pd.DataFrame(np.around(values, decimals=decimals), copy=True)

    @staticmethod
    def iterations_to_dataframe(data: Any) -> pd.DataFrame:
        """Return iteration data in the tabular form used by line plots."""
        return pd.DataFrame(data=data, copy=True)

    @staticmethod
    def aggregate_values(
        data: Any,
        map_size: Sequence[int],
        agg_axes: Sequence[int],
    ) -> np.ndarray:
        """Average numeric values over axes from the original map shape.

        Axes are normalized against the original reshaped array and reduced in
        one operation. Their order therefore does not affect the result. Empty
        aggregation axes return a copy of the reshaped values. Duplicate axes
        are rejected instead of being interpreted after an earlier reduction.

        Categorical arrays are rejected because averaging labels or metadata
        has no defined policy meaning.
        """
        values = np.asarray(data).reshape(tuple(map_size))
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("aggregate_values requires numeric measurement data")

        normalized_axes = []
        for axis in agg_axes:
            axis = index(axis)
            if axis < 0:
                axis += values.ndim
            if not 0 <= axis < values.ndim:
                raise AxisError(axis, ndim=values.ndim)
            normalized_axes.append(axis)

        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("agg_axes must not contain duplicate axes")
        if not normalized_axes:
            return values.copy()

        return np.mean(values, axis=tuple(normalized_axes))

    @staticmethod
    def values_heat_map(
        data: Any,
        title: str,
        size: Sequence[int],
        show: bool = True,
        ax: Axes | None = None,
    ) -> Axes:
        """Render state values on an explicit axes and return it.

        When ax is omitted, a new axes is created. Rendering always targets
        that explicit axes and does not depend on pyplot's current axes.
        """
        frame = Plots.values_to_dataframe(data, size)
        target = Plots._axes(ax)
        sns.heatmap(frame, annot=True, ax=target)
        target.set_title(title)
        Plots._show(show)
        return target

    @staticmethod
    def v_iters_plot(
        data: Any,
        title: str,
        show: bool = True,
        ax: Axes | None = None,
    ) -> Axes:
        """Render one or more value traces on an axes and return it."""
        frame = Plots.iterations_to_dataframe(data)
        target = Plots._axes(ax)
        Plots._apply_whitegrid(target)
        sns.lineplot(data=frame, legend=None, ax=target)
        target.set_title(title)
        Plots._show(show)
        return target

    @staticmethod
    def get_values_agg_axis_means(
        pi: Mapping[int, int] | Sequence[int],
        val_max: Any,
        map_size: Sequence[int],
        agg_axes: Sequence[int],
    ) -> np.ndarray:
        """Return numeric value means over axes from the original map shape.

        pi remains in the signature for compatibility with existing calls.
        Policy labels are categorical and are intentionally not involved in the
        aggregation. Use get_policy_map to transform policy labels.
        """
        del pi
        return Plots.aggregate_values(val_max, map_size, agg_axes)

    # modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def get_policy_map(
        pi: Mapping[int, int] | Sequence[int],
        val_max: Any,
        actions: Mapping[int, Any] | Sequence[Any] | None,
        map_size: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return reshaped values and full, untruncated policy labels."""
        values = np.asarray(val_max)
        mapped_values = values.reshape(tuple(map_size))
        labels = []
        for state in range(values.size):
            action = pi[state]
            label = actions[action] if actions is not None else action
            labels.append(str(label))

        policy_map = np.asarray(labels, dtype=object).reshape(tuple(map_size))
        return mapped_values, policy_map

    # modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def plot_policy(
        val_max: Any,
        directions: Any,
        map_size: Sequence[int],
        title: str,
        show: bool = True,
        ax: Axes | None = None,
    ) -> Axes:
        """Render a policy on an axes and return it."""
        del map_size
        target = Plots._axes(ax)
        sns.heatmap(
            val_max,
            annot=directions,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
            ax=target,
        )
        target.set_title(title)
        Plots._show(show)
        return target

    @staticmethod
    def _axes(ax: Axes | None) -> Axes:
        if ax is not None:
            return ax
        previous_figure = plt.gcf() if plt.get_fignums() else None
        previous_axes = (
            previous_figure.gca()
            if previous_figure is not None and previous_figure.axes
            else None
        )
        try:
            _, target = plt.subplots()
        finally:
            if previous_figure is not None:
                plt.figure(previous_figure.number)
                if previous_axes is not None:
                    plt.sca(previous_axes)
        return target

    @staticmethod
    def _apply_whitegrid(ax: Axes) -> None:
        style = sns.axes_style("whitegrid")
        ax.set_facecolor(style["axes.facecolor"])
        ax.set_axisbelow(True)
        ax.grid(
            True,
            color=style["grid.color"],
            linestyle=style["grid.linestyle"],
            linewidth=0.8,
        )
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(style[f"axes.spines.{side}"])

    @staticmethod
    def _show(show: bool) -> None:
        if show:
            plt.show()
