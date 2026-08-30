r"""
This module contains shared bettermdptools utility functions and classes.

## Key Components

- **Policy Evaluation**: Simulate a learned policy with `TestEnv`.
- **Plotting**: Compose pure value and policy transformations with explicit
  Matplotlib axes rendering.
- **Callbacks**: Observe typed episode and transition contexts during
  model-free training while retaining legacy hook signatures.
- **Seeding**: Seed global random generators on a best-effort basis.
"""
