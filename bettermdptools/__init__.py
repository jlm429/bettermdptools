r"""
# bettermdptools

Welcome to **bettermdptools**, a library for working with Markov Decision
Processes (MDPs) and reinforcement learning environments. It provides tools for
model-based and model-free algorithms, environment modeling, and visualization.

## Key Features

### Reinforcement Learning Algorithms
bettermdptools includes model-free algorithms such as Q-Learning and SARSA.
They work with Gymnasium environments that have single discrete-valued state
spaces. Other state spaces can be converted with a callable.

### Planning Algorithms
The library also provides model-based planning algorithms such as Value
Iteration and Policy Iteration for MDPs with known transitions and rewards.

### Environment Models
bettermdptools includes an exact context-aware Blackjack model and discretized
models for CartPole, Acrobot, and Pendulum. These adapters allow tabular
learning and planning algorithms to work with Gymnasium environments that do
not expose a native tabular transition model.

### Visualization Tools
Plotting utilities generate heatmaps, line plots, and other visualizations for
analyzing learning progress and performance.

## Getting Started

### Installation

Install bettermdptools from PyPI:

```bash
pip install bettermdptools
```

Source and contributor documentation are available in the GitHub repository:

https://github.com/jlm429/bettermdptools

## Modules

- [Utils](./bettermdptools/utils.html): Shared utility functions and classes.
- [Envs](./bettermdptools/envs.html): Environment wrappers and models.
- [Algorithms](./bettermdptools/algorithms.html): Learning and planning
  algorithms.
- [Experiments](./bettermdptools/experiments.html): High-level experiment and
  optional Optuna entrypoints.
"""
