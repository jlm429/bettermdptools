# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-08-29

### Added

- Exposed `RunResult` and the optional Optuna entrypoints consistently from the
  experiments package. Install Optuna support with `bettermdptools[optuna]`.
- Added generated API documentation for the high-level experiment workflows.

### Changed

- Breaking: migrated the supported runtime from Gymnasium 0.26 to Gymnasium
  1.3. Environments and wrappers now follow the two-value `reset` and five-value
  `step` contracts, use current environment IDs, and read native transition
  models from wrapped or unwrapped environments as appropriate.
- Breaking: formalized support for Python 3.10 through 3.12, NumPy
  `>=1.26,<2`, and bounded runtime dependencies. Python 3.13 and NumPy 2 remain
  unsupported.
- Replaced the bundled Blackjack transition pickle with an exact,
  context-aware model that distinguishes natural blackjack from later soft 21
  states.
- Moved package metadata and builds to the PEP 621 `[project]` table backed by
  Poetry, with one authoritative version declaration and an optional Optuna
  extra.

### Fixed

- Corrected termination and truncation handling across Q-learning, SARSA,
  experiment workflows, policy evaluation, and seeded resets. Time-limit
  truncation now preserves TD bootstrapping while true termination stops it.
- Corrected value and policy iteration convergence bounds and tie-breaking, and
  made short learning-rate schedules finite and validated.
- Corrected Blackjack, CartPole, Acrobot, and Pendulum transition, reward,
  observation-indexing, discretization-boundary, and model-loading behavior.
- Hardened Pendulum model caching against worker failures, incomplete or
  unreadable files, concurrent writes, and interactive runtimes where worker
  processes cannot be started safely.
- Preserved modeled wrapper semantics during rendered policy evaluation,
  including array rendering, environment ownership, and close behavior.
- Closed experiment-owned environments on success and failure, preserved
  caller-owned environments, and corrected Blackjack policy plotting.
- Updated public examples, notebooks, docstrings, and package documentation for
  the Gymnasium 1.3 and public API contracts.

[Unreleased]: https://github.com/jlm429/bettermdptools/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/jlm429/bettermdptools/compare/v0.8.6...v0.9.0
