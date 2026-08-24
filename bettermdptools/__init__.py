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
bettermdptools includes environment models for Blackjack, CartPole, and
Pendulum. Discretized models allow tabular learning and planning algorithms to
work with continuous environments.

### Visualization Tools
Plotting utilities generate heatmaps, line plots, and other visualizations for
analyzing learning progress and performance.

## Getting Started

### Installation

You can install bettermdptools via pip or by cloning the GitHub repository:

https://github.com/jlm429/bettermdptools

## Modules

- [Utils](./bettermdptools/utils.html): Shared utility functions and classes.
- [Envs](./bettermdptools/envs.html): Environment wrappers and models.
- [Algorithms](./bettermdptools/algorithms.html): Learning and planning
  algorithms.
"""
