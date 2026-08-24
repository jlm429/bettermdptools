r"""
# bettermdptools

Welcome to **bettermdptools**, a library for working with Markov Decision
Processes (MDPs) and reinforcement learning environments. It provides
model-based and model-free algorithms plus environment modeling and
visualization utilities.

## Key Features

### Reinforcement Learning Algorithms
bettermdptools implements model-free algorithms such as Q-Learning and SARSA.
They work with Gymnasium environments that have a single discrete state space.
Other state spaces can be converted with a callable.

### Planning Algorithms
The library also provides Value Iteration and Policy Iteration for MDPs with
known transition probabilities and rewards.

### Environment Models
bettermdptools provides environment models for Blackjack, CartPole, and
Pendulum, including discretized forms for tabular algorithms.

### Visualization Tools
Plotting utilities produce heatmaps, line plots, and other visualizations for
tracking agent performance.

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
