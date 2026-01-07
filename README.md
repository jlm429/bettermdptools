![PyPI](https://img.shields.io/pypi/v/bettermdptools.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/bettermdptools.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/lint-ruff-blue.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/WH9eaoZnQRJ8SGFDrvqQAd/HURrQDZ2vzVYyU2QhPL29y/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/WH9eaoZnQRJ8SGFDrvqQAd/HURrQDZ2vzVYyU2QhPL29y/tree/master)
# bettermdptools

Bettermdptools is a lightweight toolkit for working with **Gymnasium** environments using classic **planning** and **tabular reinforcement learning** methods.

It is designed to help users get up and running quickly, explore standard RL algorithms, and experiment with environments like FrozenLake, Taxi, Blackjack, and CartPole without heavy framework overhead.

---

## Getting started

### Install

Install from PyPI:

```bash
pip install bettermdptools
```

---

## Quick example (FrozenLake)

Below is a minimal example using value iteration on FrozenLake:

```python
import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots

env = gym.make("FrozenLake8x8-v1", render_mode=None)

V, V_track, pi = Planner(env.P).value_iteration()

Plots.values_heat_map(
    V,
    title="FrozenLake Value Iteration - State Values",
    size=(8, 8),
)
```
![grid_state_values](https://user-images.githubusercontent.com/10093986/211906047-bc13956b-b8e6-411d-ae68-7a3eb5f2ad32.PNG)
---

## Example notebooks

The fastest way to explore the library is through the example notebooks in the `examples/` directory.

### Core environments
- `examples/frozen_lake.ipynb`  
  Planning and tabular RL on FrozenLake

- `examples/blackjack.ipynb`  
  Q-learning on Blackjack

- `examples/cartpole.ipynb`  
  Discretized CartPole using a wrapper for tabular methods

### Experiments API
- `examples/experiments_demo.ipynb`  
  Demonstrates the optional `bettermdptools.experiments.run(...)` entrypoint across:
  - FrozenLake (VI and Q-learning)
  - Blackjack (Q-learning)
  - CartPole (discretized)

### Utilities and plotting
- `examples/plots.ipynb`  
  Visualization helpers

- `examples/other_utilities.ipynb`  
  Miscellaneous helpers

These notebooks are intended as examples and starting points, not benchmarks.

---

## Experiments API (optional)

Bettermdptools includes an optional high-level experiments entrypoint that wires together:
- environment creation
- transition model handling
- algorithm dispatch
- optional policy evaluation

Primary entrypoint:

```python
from bettermdptools.experiments import run
```

See:
- `examples/experiments_demo.ipynb`
- `docs/api/entrypoints/experiments.md`

---

## pygame installation issues (Python 3.11+)

If you encounter errors installing `pygame` on Python 3.11 or newer, try:

```bash
pip install pygame --pre
```

See:
- https://stackoverflow.com/questions/74188013/python-pygame-not-installing

---

## Development and tooling

### Dependency management

The project supports standard `pip` workflows. For development, using **Poetry** is recommended to ensure reproducible environments.

Example:

```bash
poetry install
poetry shell
```

### Code quality

The codebase uses the following tools during development:

- **ruff** for fast linting
- **black** for code formatting

Typical usage:

```bash
ruff check .
black .
```

---

## Documentation

API documentation is generated using **pdoc** (not pdoc3) and lives in the `docs/` directory.

To regenerate documentation locally:

```bash
pdoc --include-undocumented -d numpy -t docs-templates --output-dir docs bettermdptools
```

---

## Contributing

Pull requests are welcome.

Guidelines:
- Use numpy-style docstrings
- Add or update tests when introducing new functionality
- Prefer explicit, readable code over clever or showy abstractions
- Format code with **black** and check linting with **ruff** before submitting
- Keep public APIs stable and avoid breaking changes unless clearly justified
- Prefer small, focused pull requests with a limited number of file changes

Basic workflow:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---