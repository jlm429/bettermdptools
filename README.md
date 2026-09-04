[![PyPI](https://img.shields.io/pypi/v/bettermdptools.svg)](https://pypi.org/project/bettermdptools/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bettermdptools.svg)](https://pypi.org/project/bettermdptools/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/jlm429/bettermdptools/blob/master/LICENSE)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/HURrQDZ2vzVYyU2QhPL29y/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/HURrQDZ2vzVYyU2QhPL29y/tree/master)

<p align="center">
  <img src="https://raw.githubusercontent.com/jlm429/bettermdptools/master/docs/assets/bettermdptools-banner.png"
       alt="BetterMDPTools reinforcement learning gridworld"
       width="100%">
</p>

# BetterMDPTools

Learn how classic Markov decision process algorithms work by solving,
experimenting with, and visualizing tabular Gymnasium environments.
BetterMDPTools gives you direct access to value iteration, policy iteration,
Q-learning, and SARSA, with inspectable NumPy results and composable plotting
tools.

## See values and policy together

Turn a solved FrozenLake environment into a value map and readable policy with
the public plotting API:

<p align="center">
  <img src="https://raw.githubusercontent.com/jlm429/bettermdptools/c8801b83496b3257695a93c4b01dfde70557b940/docs/assets/frozen-lake-value-policy.png"
       alt="FrozenLake state values and policy plotted side by side"
       width="900">
</p>

Plot preparation is separate from rendering. Every renderer draws on explicit,
caller-owned Matplotlib axes, so plots compose naturally in notebooks,
applications, and reports.

## What you can explore

| Area | Included tools |
| --- | --- |
| Planning | `Planner.value_iteration`, vectorized value iteration, and `Planner.policy_iteration` for tabular transition models |
| Tabular reinforcement learning | `RL.q_learning` and `RL.sarsa` for learning through Gymnasium interactions |
| Environments | Native discrete Gymnasium environments plus model and discretization wrappers for Blackjack, CartPole, Acrobot, and Pendulum |
| Experiments | A reusable `run` entrypoint, `ExperimentBuilder`, seeded evaluation, and optional Optuna search |
| Visualization | Learning curves, value and policy convergence, value heatmaps, policy grids, and combined value-policy figures |

Planning methods return state values, valid convergence history, and a policy.
Model-free methods also return action values, per-episode policy history, and
rewards, making the learning process available for analysis rather than hiding
it behind a training loop.

## Installation

BetterMDPTools supports Python 3.12 through 3.14, NumPy 2.x, and Gymnasium 1.3.
The standard installation includes planning, training, evaluation, plotting,
and non-rendering notebook workflows:

```bash
pip install bettermdptools
```

## Quick start

This example solves the built-in 8 by 8 FrozenLake transition model and plots
the values and greedy policy side by side:

```python
import gymnasium as gym
import matplotlib.pyplot as plt

from bettermdptools.algorithms.planner import Planner
from bettermdptools.plotting import plot_value_policy, prepare_policy_grid

env = gym.make("FrozenLake8x8-v1", render_mode=None)

V, V_track, pi = Planner(env.unwrapped.P).value_iteration(gamma=0.99)
grid = prepare_policy_grid(
    policy=pi,
    values=V,
    action_labels={0: "←", 1: "↓", 2: "→", 3: "↑"},
    shape=(8, 8),
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
plot_value_policy(grid, value_ax=axes[0], policy_ax=axes[1])
plt.show()
plt.close(fig)
env.close()
```

For policy iteration, replace `value_iteration(...)` with
`policy_iteration(...)`. For model-free learning, pass the environment to
`RL(env).q_learning(...)` or `RL(env).sarsa(...)`.

Gymnasium's built-in tabular transition models live on `env.unwrapped.P`.
BetterMDPTools wrappers expose their generated models through `.P` and convert
continuous or context-dependent observations into discrete state spaces.

## Examples and documentation

The [`examples/`](https://github.com/jlm429/bettermdptools/tree/master/examples)
directory contains executed notebooks with saved plots and results:

- [`frozen_lake.ipynb`](https://github.com/jlm429/bettermdptools/blob/master/examples/frozen_lake.ipynb)
  introduces planning and a value heatmap.
- [`plots.ipynb`](https://github.com/jlm429/bettermdptools/blob/master/examples/plots.ipynb)
  covers learning curves, convergence diagnostics, figure composition, and
  style controls.
- [`experiments_demo.ipynb`](https://github.com/jlm429/bettermdptools/blob/master/examples/experiments_demo.ipynb)
  runs planning and Q-learning across discrete and wrapped environments.
- [`optuna_search_examples.ipynb`](https://github.com/jlm429/bettermdptools/blob/master/examples/optuna_search_examples.ipynb)
  tunes Q-learning, SARSA, and value iteration.

Detailed guides are available for the
[plotting API](https://github.com/jlm429/bettermdptools/blob/master/docs/api/plotting_api.md),
[experiment API](https://github.com/jlm429/bettermdptools/blob/master/docs/api/experiments_api.md),
and [Optuna integration](https://github.com/jlm429/bettermdptools/blob/master/docs/api/optuna_search_api.md).
The generated [Python API reference](https://jlm429.github.io/bettermdptools/bettermdptools.html)
documents every public module.

## Optional rendering and Optuna

Install local Gymnasium rendering support, Optuna search, or both:

```bash
pip install "bettermdptools[rendering]"
pip install "bettermdptools[optuna]"
pip install "bettermdptools[rendering,optuna]"
```

The `rendering` extra installs `pygame-ce>=2.5.5,<3`, which supplies the
`pygame` interface used by Gymnasium. Classic `pygame` is not supported and
should not be installed alongside `pygame-ce`. Rendering is not supported on
Google Colab, but core non-rendering workflows work there without this extra.

The `optuna` extra installs `optuna>=4.6,<5`. It is only needed when calling
`bettermdptools.experiments.optimize`.

The package requires Python `>=3.12,<3.15`, NumPy `>=2,<3`, Gymnasium
`>=1.3,<1.4`, and Matplotlib `>=3.8,<4`.

## Development and contributing

Poetry owns dependency resolution, builds, tests, and documentation tooling:

```bash
poetry install --with docs
poetry run pytest -q
poetry run ruff check .
poetry run black --check .
poetry check --lock
poetry build
```

See [`CONTRIBUTING.md`](https://github.com/jlm429/bettermdptools/blob/master/CONTRIBUTING.md)
for the complete contributor workflow.

## License

BetterMDPTools is distributed under the
[BSD 3-Clause License](https://github.com/jlm429/bettermdptools/blob/master/LICENSE).
