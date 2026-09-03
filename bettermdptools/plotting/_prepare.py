"""Pure preparation and validation for plotting data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from operator import index
from typing import Literal

import numpy as np
from numpy.exceptions import AxisError
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ValueGridData:
    """Numeric values and display annotations for a two-dimensional grid."""

    values: NDArray[np.floating]
    annotations: NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class PolicyGridData:
    """Numeric state values and complete action labels for a policy grid."""

    values: NDArray[np.floating]
    actions: NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class LearningCurveData:
    """Episode rewards and repeated-run summary statistics.

    Arrays use a ``(runs, episodes)`` convention. One-dimensional input is one
    run. Smoothing is applied independently within each run before the center
    and interval are calculated across runs.
    """

    episodes: NDArray[np.integer]
    rewards: NDArray[np.floating]
    smoothed: NDArray[np.floating]
    center: NDArray[np.floating]
    lower: NDArray[np.floating] | None
    upper: NDArray[np.floating] | None
    window: int
    center_statistic: Literal["mean", "median"]
    interval: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class ConvergenceData:
    """Differences between consecutive value and policy history entries."""

    iterations: NDArray[np.integer]
    value_delta: NDArray[np.floating]
    policy_changes: NDArray[np.integer] | None
    valid_length: int
    value_statistic: Literal["max", "mean"]


def _grid_shape(shape: Sequence[int]) -> tuple[int, int]:
    try:
        dimensions = tuple(index(dimension) for dimension in shape)
    except TypeError as error:
        raise TypeError("shape must contain exactly two integers") from error
    if len(dimensions) != 2 or any(dimension <= 0 for dimension in dimensions):
        raise ValueError("shape must contain exactly two positive integers")
    return dimensions


def _numeric_array(data: ArrayLike, name: str) -> NDArray[np.floating]:
    values = np.asarray(data)
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise TypeError(f"{name} must contain real numeric values")
    values = np.array(values, dtype=float, copy=True)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinite values")
    return values


def prepare_value_grid(
    values: ArrayLike,
    shape: Sequence[int],
    *,
    decimals: int = 2,
) -> ValueGridData:
    """Prepare numeric values and formatted annotations for a grid.

    ``NaN`` values are retained so renderers can mask missing states. Infinite
    values are rejected because they do not define a useful color scale.
    Numeric values are not rounded, so annotation formatting never changes the
    values used for color normalization.
    """
    grid_shape = _grid_shape(shape)
    if isinstance(decimals, bool) or not isinstance(decimals, Integral) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    data = _numeric_array(values, "values")
    if data.size != int(np.prod(grid_shape)):
        raise ValueError(
            f"shape {grid_shape} requires {int(np.prod(grid_shape))} values, "
            f"but received {data.size}"
        )
    grid = data.reshape(grid_shape)
    annotations = np.empty(grid_shape, dtype=object)
    formatter = f"{{:.{int(decimals)}f}}".format
    for cell, value in np.ndenumerate(grid):
        annotations[cell] = "" if np.isnan(value) else formatter(value)
    return ValueGridData(values=grid, annotations=annotations)


def aggregate_values(
    values: ArrayLike,
    shape: Sequence[int],
    axes: Sequence[int],
) -> NDArray[np.floating]:
    """Average numeric values over axes from the original source shape.

    Negative axes are supported. Duplicate axes are rejected, and an empty
    axis sequence returns a copy of the reshaped input.
    """
    try:
        source_shape = tuple(index(dimension) for dimension in shape)
    except TypeError as error:
        raise TypeError("shape must contain integers") from error
    if not source_shape or any(dimension <= 0 for dimension in source_shape):
        raise ValueError("shape must contain positive integers")
    data = _numeric_array(values, "values")
    if data.size != int(np.prod(source_shape)):
        raise ValueError(
            f"shape {source_shape} requires {int(np.prod(source_shape))} values, "
            f"but received {data.size}"
        )
    data = data.reshape(source_shape)

    normalized_axes: list[int] = []
    for raw_axis in axes:
        axis = index(raw_axis)
        if axis < 0:
            axis += data.ndim
        if not 0 <= axis < data.ndim:
            raise AxisError(axis, ndim=data.ndim)
        normalized_axes.append(axis)
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("axes must not contain duplicates")
    if not normalized_axes:
        return data.copy()
    return np.mean(data, axis=tuple(normalized_axes))


def prepare_policy_grid(
    policy: Mapping[int, int] | Sequence[int],
    values: ArrayLike,
    action_labels: Mapping[int, object] | Sequence[object] | None,
    shape: Sequence[int],
) -> PolicyGridData:
    """Prepare a complete state-value and action-label policy grid.

    The policy must define every state from zero through ``values.size - 1``.
    Actions must be integral indices. When ``action_labels`` is omitted, the
    numeric actions are converted to strings.
    """
    value_data = prepare_value_grid(values, shape)
    flat_values = value_data.values.ravel()
    labels: list[str] = []
    for state in range(flat_values.size):
        try:
            action = policy[state]
        except (IndexError, KeyError) as error:
            raise ValueError(f"policy is missing state {state}") from error
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError(f"policy action for state {state} must be an integer")
        action = int(action)
        if action < 0:
            raise ValueError(f"policy action for state {state} must be non-negative")
        if action_labels is None:
            label = action
        else:
            try:
                label = action_labels[action]
            except (IndexError, KeyError) as error:
                raise ValueError(f"action_labels is missing action {action}") from error
        labels.append(str(label))
    actions = np.asarray(labels, dtype=object).reshape(value_data.values.shape)
    return PolicyGridData(values=value_data.values, actions=actions)


def _rolling_mean(values: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    totals = np.cumsum(values, axis=1, dtype=float)
    totals[:, window:] -= totals[:, :-window]
    divisors = np.minimum(np.arange(1, values.shape[1] + 1), window)
    return totals / divisors


def prepare_learning_curve(
    rewards: ArrayLike,
    *,
    window: int = 1,
    center: Literal["mean", "median"] = "mean",
    interval: tuple[float, float] | None = (0.1, 0.9),
) -> LearningCurveData:
    """Prepare episode rewards and an optional repeated-run interval.

    Parameters
    ----------
    rewards
        One run with shape ``(episodes,)`` or repeated runs with shape
        ``(runs, episodes)``. Runs must have equal episode counts.
    window
        Trailing mean window applied independently within each run. Initial
        points use all observations available so the output length is stable.
    center
        Mean or median across runs after smoothing.
    interval
        Lower and upper quantiles in ``[0, 1]``. The band is produced only for
        repeated runs. Pass ``None`` to disable it.
    """
    data = _numeric_array(rewards, "rewards")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim != 2:
        raise ValueError("rewards must have shape (episodes,) or (runs, episodes)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("rewards must contain at least one run and one episode")
    if np.isnan(data).any():
        raise ValueError("rewards must not contain NaN values")
    if isinstance(window, bool) or not isinstance(window, Integral) or window < 1:
        raise ValueError("window must be a positive integer")
    if center not in {"mean", "median"}:
        raise ValueError("center must be 'mean' or 'median'")
    if interval is not None:
        if len(interval) != 2:
            raise ValueError("interval must contain lower and upper quantiles")
        lower_quantile, upper_quantile = interval
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("interval quantiles must satisfy 0 <= lower < upper <= 1")

    smoothed = _rolling_mean(data, int(window))
    reducer = np.mean if center == "mean" else np.median
    center_values = reducer(smoothed, axis=0)
    lower = upper = None
    if data.shape[0] > 1 and interval is not None:
        lower, upper = np.quantile(smoothed, interval, axis=0)
    return LearningCurveData(
        episodes=np.arange(1, data.shape[1] + 1),
        rewards=data,
        smoothed=smoothed,
        center=np.asarray(center_values),
        lower=None if lower is None else np.asarray(lower),
        upper=None if upper is None else np.asarray(upper),
        window=int(window),
        center_statistic=center,
        interval=interval,
    )


def prepare_convergence(
    value_history: ArrayLike,
    *,
    policy_history: ArrayLike | None = None,
    valid_length: int | None = None,
    value_statistic: Literal["max", "mean"] = "max",
) -> ConvergenceData:
    """Prepare value deltas and policy-change counts between history entries.

    The first dimension is time and every remaining dimension is flattened for
    comparison. ``valid_length`` is the sole validity control. When omitted,
    every supplied row is treated as valid. No zero-valued row is ever trimmed
    or interpreted as padding.
    """
    values = _numeric_array(value_history, "value_history")
    if values.ndim < 2:
        raise ValueError("value_history must have time plus at least one value axis")
    if np.isnan(values).any():
        raise ValueError("value_history must not contain NaN values")
    if valid_length is None:
        length = values.shape[0]
    else:
        if (
            isinstance(valid_length, bool)
            or not isinstance(valid_length, Integral)
            or not 2 <= valid_length <= values.shape[0]
        ):
            raise ValueError(
                "valid_length must be an integer from 2 through the history length"
            )
        length = int(valid_length)
    if length < 2:
        raise ValueError("value_history must contain at least two valid entries")
    if value_statistic not in {"max", "mean"}:
        raise ValueError("value_statistic must be 'max' or 'mean'")

    differences = np.abs(np.diff(values[:length], axis=0)).reshape(length - 1, -1)
    reducer = np.max if value_statistic == "max" else np.mean
    value_delta = reducer(differences, axis=1)

    policy_changes = None
    if policy_history is not None:
        policies = np.asarray(policy_history)
        if policies.ndim < 2:
            raise ValueError(
                "policy_history must have time plus at least one policy axis"
            )
        if policies.shape[0] < length:
            raise ValueError("policy_history is shorter than valid_length")
        flattened = policies[:length].reshape(length, -1)
        policy_changes = np.count_nonzero(flattened[1:] != flattened[:-1], axis=1)

    return ConvergenceData(
        iterations=np.arange(1, length),
        value_delta=np.asarray(value_delta),
        policy_changes=None if policy_changes is None else np.asarray(policy_changes),
        valid_length=length,
        value_statistic=value_statistic,
    )
