r"""
# bettermdptools

Welcome to **bettermdptools**, a comprehensive library designed to facilitate working with Markov Decision Processes (MDPs) and reinforcement learning environments. This library is built to provide robust tools for both model-based and model-free reinforcement learning algorithms, along with utilities for environment modeling, visualization, and more.

## Key Features

### Reinforcement Learning Algorithms
bettermdptools includes implementations of popular model-free reinforcement learning algorithms such as Q-Learning and SARSA. These algorithms are designed to work seamlessly with any gymnasium environments that have single discrete valued state spaces. For environments that do not fit this format, a lambda function can be used to convert state spaces accordingly.

### Planning Algorithms
The library also provides model-based planning algorithms like Value Iteration and Policy Iteration. These algorithms are essential for solving MDPs where the transition probabilities and rewards are known.

### Environment Models
bettermdptools comes with pre-built environment models for popular problems such as Blackjack, CartPole, and Pendulum. These models include discretized versions of the environments, making it easier to apply traditional reinforcement learning and planning algorithms.

### Visualization Tools
To help you better understand and analyze the performance of your algorithms, bettermdptools includes a variety of plotting utilities. These tools can generate heatmaps, line plots, and other visualizations to track the progress and performance of your learning agents.

## Getting Started

### Installation

You can install bettermdptools via pip or by cloning the GitHub repository:

https://github.com/jlm429/bettermdptools

## Modules

- [Utils](./bettermdptools/utils.html): Utility functions and classes used across the library.
- [Envs](./bettermdptools/envs.html): Environment wrappers and models for various reinforcement learning environments.
- [Algorithms](./bettermdptools/algorithms.html): Implementations of various reinforcement learning and planning algorithms.
"""
