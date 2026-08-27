[![PyPI](https://img.shields.io/pypi/v/bettermdptools.svg)](https://pypi.org/project/bettermdptools/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bettermdptools.svg)](https://pypi.org/project/bettermdptools/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/jlm429/bettermdptools/blob/master/LICENSE)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/HURrQDZ2vzVYyU2QhPL29y/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/HURrQDZ2vzVYyU2QhPL29y/tree/master)

# bettermdptools

bettermdptools provides classic planning and tabular reinforcement learning
algorithms for [Gymnasium](https://gymnasium.farama.org/) environments. It
includes value iteration, policy iteration, Q-learning, SARSA, environment
adapters, experiment entrypoints, and plotting utilities.

## Installation

bettermdptools supports Python 3.10 and newer:

```bash
pip install bettermdptools
```

Install the optional Optuna integration with:

```bash
pip install "bettermdptools[optuna]"
```

## Quick example

The transition model for Gymnasium's built-in discrete environments is stored
on the unwrapped environment:

```python
import gymnasium as gym

from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots

env = gym.make("FrozenLake8x8-v1", render_mode=None)
V, V_track, pi = Planner(env.unwrapped.P).value_iteration(gamma=0.99)

Plots.values_heat_map(V, title="State Values", size=(8, 8))
env.close()
```

bettermdptools wrappers expose generated tabular models through their own `.P`
property. The Blackjack wrapper uses a context-aware exact representation.
CartPole, Acrobot, and Pendulum use discretized models.

## Examples and API documentation

User-facing notebooks live in
[`examples/`](https://github.com/jlm429/bettermdptools/tree/master/examples).
The high-level
experiment and optional Optuna APIs are documented in:

- [`docs/api/experiments_api.md`](https://github.com/jlm429/bettermdptools/blob/master/docs/api/experiments_api.md)
- [`docs/api/optuna_search_api.md`](https://github.com/jlm429/bettermdptools/blob/master/docs/api/optuna_search_api.md)

The generated Python API reference starts at
[`bettermdptools.html`](https://jlm429.github.io/bettermdptools/bettermdptools.html).
Python docstrings are the source for that reference.

## Development

Poetry is the source of truth for dependencies, builds, and documentation
tooling:

```bash
poetry install --with docs
poetry run pytest -q
poetry run ruff check .
poetry run black --check .
poetry run pdoc bettermdptools -o docs
```

See
[`CONTRIBUTING.md`](https://github.com/jlm429/bettermdptools/blob/master/CONTRIBUTING.md)
for the complete contributor workflow.

## License

bettermdptools is distributed under the BSD 3-Clause License. See
[`LICENSE`](https://github.com/jlm429/bettermdptools/blob/master/LICENSE).
