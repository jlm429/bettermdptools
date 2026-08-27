r"""
This module contains reinforcement learning environment wrappers and models.

These wrappers and models expose discrete observations, actions, and transition
dictionaries for bettermdptools algorithms. Blackjack uses an exact model that
distinguishes natural blackjack from later soft 21 states. CartPole, Acrobot,
and Pendulum use discretized models.

## Key Components

- **Environment Wrappers**: Classes that adapt existing environments.
- **Environment Models**: Exact or discretized transition and reward models for
  tabular algorithms.
"""
