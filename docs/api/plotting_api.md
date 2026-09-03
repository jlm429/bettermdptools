# Plotting API

`bettermdptools.plotting` provides pure preparation functions and Matplotlib
renderers for algorithm results. Preparation validates and reshapes arrays
without importing Matplotlib. Every renderer requires explicit caller-owned
`Axes`, draws on those axes, and returns the same axes.

Renderers never call `show`, save, close, resize a figure, or apply
figure-level layout. The caller decides layout and owns the complete figure
lifecycle.

## Figure ownership

Create figures with Matplotlib, pass their axes to BetterMDPTools, save before
a blocking `plt.show()`, and close figures explicitly in batch or long-running
workflows:

```python
import matplotlib.pyplot as plt

from bettermdptools.plotting import (
    plot_value_policy,
    prepare_policy_grid,
)

data = prepare_policy_grid(
    policy=pi,
    values=V,
    action_labels={0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"},
    shape=(8, 8),
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
plot_value_policy(data, value_ax=axes[0], policy_ax=axes[1])
fig.savefig("frozen-lake.png", dpi=150)
plt.show()
plt.close(fig)
```

Use `fig.suptitle(...)` for a figure title. BetterMDPTools renderer titles are
axes titles. When several panels need one color scale, supply a shared
`matplotlib.colors.Normalize`, allocate a `cbar_ax`, and disable redundant
colorbars.

## Episode reward learning curves

`prepare_learning_curve` accepts either one reward history with shape
`(episodes,)` or repeated runs with shape `(runs, episodes)`. Repeated runs
must contain the same number of episodes. A trailing mean is computed inside
each run first. The requested mean or median and optional quantile interval are
then computed across runs.

```python
fig, ax = plt.subplots(layout="constrained")
curve = prepare_learning_curve(reward_runs, window=100, interval=(0.1, 0.9))
plot_learning_curve(curve, ax=ax, title="Q-learning rewards")
```

The interval describes variation across independent runs. A one-run rolling
line is smoothing, not statistical uncertainty, so a one-run curve has no
interval band. The renderer applies a white-grid style to its supplied axes
without changing process-global Matplotlib or seaborn configuration.

The low-level RL return and `RunResult.train` use the same preparation path:

```python
# Direct RL tuple
Q, V, pi, Q_track, pi_track, rewards = RL(env).q_learning()
direct_curve = prepare_learning_curve(rewards, window=100)

# High-level experiment result
out = run(algo="q_learning", env_id="FrozenLake-v1")
experiment_curve = prepare_learning_curve(out.train["rewards"], window=100)
```

## Value and policy convergence

`prepare_convergence` treats the first dimension as time. It accepts common
histories such as `(iterations, states)` values and
`(episodes, states, actions)` Q values. Every remaining dimension is flattened
before the maximum or mean absolute change is calculated. Optional policy
history uses `(iterations, ...)` and reports the number of states whose action
changed between consecutive entries.

```python
convergence = prepare_convergence(Q_track, policy_history=pi_track)
fig, axes = plt.subplots(1, 2, layout="constrained")
plot_convergence(
    convergence,
    value_ax=axes[0],
    policy_ax=axes[1],
    title="Training convergence",
)
```

No preparation function infers validity from numeric values. In particular,
trailing zero rows are never removed because an all-zero estimate can be valid.
Q-learning and SARSA record one valid history entry per episode, so their full
histories are valid.

Planner histories preserve their historical fixed allocation. Request exact
metadata and pass its valid prefix explicitly:

```python
V, V_track, pi, metadata = Planner(P).value_iteration(
    return_metadata=True
)
convergence = prepare_convergence(
    V_track,
    valid_length=metadata.history_length,
)
```

`PlanningMetadata.history_length` includes the initial history row. Existing
planner calls still return the historical three-item tuple unless
`return_metadata=True` is requested.

## Value and policy grids

`prepare_value_grid(values, shape)` accepts flat or already-shaped numeric
values whose element count exactly matches a positive two-dimensional shape.
It retains `NaN` as missing data, rejects infinity, and formats annotations
without rounding the numeric data used for color normalization.

`prepare_policy_grid(policy, values, action_labels, shape)` additionally
requires a policy action for every integer state from zero through
`values.size - 1`. Policies may be mappings or indexable sequences. Actions
must be integer indices, action label coverage must be complete, and full
multi-character labels are preserved.

Use `aggregate_values(values, shape, axes)` before grid preparation when a
high-dimensional state space needs numeric aggregation. Axes refer to the
original shape and are reduced simultaneously, so their order does not change
the result.

Blackjack has one terminal bust sink beyond its `(29, 10)` decision surface.
Select decision states explicitly instead of asking plotting code to infer
environment semantics:

```python
decision_values = V[:-1]
decision_policy = {
    state: pi[state] for state in range(len(decision_values))
}
data = prepare_policy_grid(
    decision_policy,
    decision_values,
    {0: "STICK", 1: "HIT"},
    (29, 10),
)
```

## Migrating from `Plots`

The unreleased API replaces `bettermdptools.utils.plots.Plots`. Use
`prepare_value_grid` plus `plot_value_heatmap` instead of `values_heat_map`,
`prepare_policy_grid` plus `plot_policy_grid` instead of `get_policy_map` and
`plot_policy`, and the learning or convergence APIs instead of the ambiguous
`v_iters_plot`. Call `plt.show()` explicitly after every renderer needed for a
figure has run.

## Examples

- [`../../examples/plots.ipynb`](../../examples/plots.ipynb) covers the three
  primary plotting workflows and public style controls.
- [`../../examples/experiments_demo.ipynb`](../../examples/experiments_demo.ipynb)
  plots rewards directly from `RunResult.train`.
- [`../../examples/optuna_search_examples.ipynb`](../../examples/optuna_search_examples.ipynb)
  uses the best run while leaving Optuna-native study visualization to Optuna.
