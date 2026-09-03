"""Axes-owned Matplotlib renderers for prepared plotting data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import matplotlib.axes
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from ._prepare import (
    ConvergenceData,
    LearningCurveData,
    PolicyGridData,
    ValueGridData,
)


class ConvergenceAxes(NamedTuple):
    """Axes returned by :func:`plot_convergence`."""

    values: matplotlib.axes.Axes
    policy: matplotlib.axes.Axes | None


class ValuePolicyAxes(NamedTuple):
    """Axes returned by :func:`plot_value_policy`."""

    values: matplotlib.axes.Axes
    policy: matplotlib.axes.Axes


def _colorbar_kwargs(label: str | None) -> dict[str, str] | None:
    return None if label is None else {"label": label}


def _apply_whitegrid(ax: Axes) -> None:
    """Apply seaborn's white-grid styling to one axes without global state."""
    style = sns.axes_style("whitegrid")
    ax.set_facecolor(style["axes.facecolor"])
    ax.set_axisbelow(style["axes.axisbelow"])
    ax.grid(
        True,
        color=style["grid.color"],
        linestyle=style["grid.linestyle"],
    )


def plot_value_heatmap(
    data: ValueGridData,
    *,
    ax: Axes,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    cbar_ax: Axes | None = None,
    value_label: str | None = "State value",
    annotate: bool = True,
    show_coordinates: bool = True,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
) -> Axes:
    """Draw a prepared value grid on and return ``ax``.

    The renderer does not show, save, close, resize, or apply figure-level
    layout. A colorbar is added to ``ax.figure`` when requested. Supply
    ``cbar_ax`` to control its exact location.
    """
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("norm cannot be combined with vmin or vmax")
    sns.heatmap(
        data.values,
        annot=data.annotations if annotate else False,
        fmt="",
        cmap=cmap,
        cbar=colorbar,
        cbar_ax=cbar_ax,
        cbar_kws=_colorbar_kwargs(value_label),
        xticklabels="auto" if show_coordinates else False,
        yticklabels="auto" if show_coordinates else False,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        annot_kws=None if annotation_kws is None else dict(annotation_kws),
        ax=ax,
    )
    if title is not None:
        ax.set_title(title)
    return ax


def plot_policy_grid(
    data: PolicyGridData,
    *,
    ax: Axes,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    cbar_ax: Axes | None = None,
    value_label: str | None = "State value",
    annotate: bool = True,
    show_coordinates: bool = True,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    linewidths: float = 0.5,
    linecolor: str = "white",
    annotation_kws: Mapping[str, Any] | None = None,
) -> Axes:
    """Draw a prepared policy grid on and return ``ax``.

    Action strings are annotations and state values control cell color. The
    renderer leaves display, saving, closing, and layout to the figure owner.
    """
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("norm cannot be combined with vmin or vmax")
    sns.heatmap(
        data.values,
        annot=data.actions if annotate else False,
        fmt="",
        cmap=cmap,
        cbar=colorbar,
        cbar_ax=cbar_ax,
        cbar_kws=_colorbar_kwargs(value_label),
        xticklabels="auto" if show_coordinates else False,
        yticklabels="auto" if show_coordinates else False,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
        linecolor=linecolor,
        annot_kws=None if annotation_kws is None else dict(annotation_kws),
        ax=ax,
    )
    if title is not None:
        ax.set_title(title)
    return ax


def plot_learning_curve(
    data: LearningCurveData,
    *,
    ax: Axes,
    title: str | None = None,
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    label: str | None = None,
    color: str | None = "C0",
    show_runs: bool = True,
    run_alpha: float = 0.15,
    interval_alpha: float = 0.2,
) -> Axes:
    """Draw episode rewards and their repeated-run summary on ``ax``.

    A white-grid style is applied only to ``ax``. Matplotlib and seaborn global
    configuration remains unchanged.
    """
    _apply_whitegrid(ax)
    if show_runs:
        for rewards in data.rewards:
            ax.plot(data.episodes, rewards, color=color, alpha=run_alpha, linewidth=0.8)
    summary_label = label
    if summary_label is None and data.rewards.shape[0] > 1:
        summary_label = data.center_statistic
    summary_line = ax.plot(
        data.episodes,
        data.center,
        color=color,
        linewidth=2,
        label=summary_label,
    )[0]
    if data.lower is not None and data.upper is not None:
        band_label = None
        if data.interval is not None:
            band_label = f"{data.interval[0]:.0%} to {data.interval[1]:.0%} across runs"
        ax.fill_between(
            data.episodes,
            data.lower,
            data.upper,
            color=summary_line.get_color(),
            alpha=interval_alpha,
            label=band_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if summary_label is not None or data.lower is not None:
        ax.legend()
    return ax


def plot_value_convergence(
    data: ConvergenceData,
    *,
    ax: Axes,
    title: str | None = None,
    xlabel: str = "Iteration",
    ylabel: str | None = None,
    color: str | None = None,
) -> Axes:
    """Draw consecutive value-history differences on and return ``ax``."""
    label = f"{data.value_statistic} absolute value change"
    ax.plot(data.iterations, data.value_delta, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or label.capitalize())
    if title is not None:
        ax.set_title(title)
    return ax


def plot_policy_convergence(
    data: ConvergenceData,
    *,
    ax: Axes,
    title: str | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "States with changed action",
    color: str | None = None,
) -> Axes:
    """Draw policy-change counts on and return ``ax``."""
    if data.policy_changes is None:
        raise ValueError("data does not contain policy history")
    ax.step(
        data.iterations,
        data.policy_changes,
        where="post",
        color=color,
        label="Policy changes",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_convergence(
    data: ConvergenceData,
    *,
    value_ax: Axes,
    policy_ax: Axes | None = None,
    title: str | None = None,
) -> ConvergenceAxes:
    """Compose value and optional policy convergence on supplied axes."""
    if data.policy_changes is not None and policy_ax is None:
        raise ValueError("policy_ax is required when data contains policy history")
    if data.policy_changes is None and policy_ax is not None:
        raise ValueError("policy_ax was supplied but data has no policy history")
    plot_value_convergence(data, ax=value_ax, title=title)
    if policy_ax is not None:
        plot_policy_convergence(data, ax=policy_ax, title=title)
    return ConvergenceAxes(values=value_ax, policy=policy_ax)


def plot_value_policy(
    data: PolicyGridData,
    *,
    value_ax: Axes,
    policy_ax: Axes,
    value_title: str | None = "State values",
    policy_title: str | None = "Policy",
    cmap: str = "viridis",
    value_colorbar: bool = True,
    policy_colorbar: bool = False,
    cbar_ax: Axes | None = None,
    value_label: str | None = "State value",
    show_coordinates: bool = True,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
) -> ValuePolicyAxes:
    """Compose matching value and policy grids on two supplied axes.

    When ``cbar_ax`` is supplied, it receives the sole enabled colorbar. An
    explicit target is ambiguous when both colorbars are enabled, so that
    combination raises ``ValueError``. Without ``cbar_ax``, Matplotlib owns
    automatic colorbar placement for each enabled panel.
    """
    if cbar_ax is not None and value_colorbar and policy_colorbar:
        raise ValueError(
            "cbar_ax cannot target both value and policy colorbars; "
            "supply separate rendering calls or omit cbar_ax"
        )
    value_cbar_ax = cbar_ax if value_colorbar else None
    policy_cbar_ax = cbar_ax if policy_colorbar else None
    annotations = np.empty(data.values.shape, dtype=object)
    for cell, value in np.ndenumerate(data.values):
        annotations[cell] = "" if np.isnan(value) else f"{value:.2f}"
    values = ValueGridData(values=data.values, annotations=annotations)
    plot_value_heatmap(
        values,
        ax=value_ax,
        title=value_title,
        cmap=cmap,
        colorbar=value_colorbar,
        cbar_ax=value_cbar_ax,
        value_label=value_label,
        show_coordinates=show_coordinates,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        annotation_kws=annotation_kws,
    )
    plot_policy_grid(
        data,
        ax=policy_ax,
        title=policy_title,
        cmap=cmap,
        colorbar=policy_colorbar,
        cbar_ax=policy_cbar_ax,
        value_label=value_label,
        show_coordinates=show_coordinates,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        annotation_kws=annotation_kws,
    )
    return ValuePolicyAxes(values=value_ax, policy=policy_ax)
