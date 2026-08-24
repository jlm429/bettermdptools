---
name: environment-model-validation
description: Validate changes to bettermdptools environment models, Gymnasium wrappers, adapters, or discretization with explicit transition and reward invariants. Use for work in bettermdptools/envs or failures caused by environment-model assumptions.
---

# Validate an environment model or adapter

Read the repository [agent guide](../../../AGENTS.md) first.

## Trace the representation

1. Inspect the affected files in `bettermdptools/envs/`, the corresponding cases in
   `tests/test_envs.py`, and the relevant notebook in `examples/`.
2. Trace a representative observation from `reset()` and `step()` through any
   discretization or conversion into the integer state consumed by `Planner` or
   `RL`.
3. Treat Gymnasium's wrapped and unwrapped attributes deliberately. Do not assume a
   wrapper exposes an internal model unless that contract is verified.

## Validate invariants

- State indices are integers in the declared range, and distinct modeled states do
  not collide unintentionally.
- Action indices match the environment action space and every modeled state has the
  expected action entries.
- Each `(state, action)` transition list has valid probabilities whose total is one
  within an appropriate numerical tolerance.
- Next-state indices are valid, rewards match environment behavior, and terminal
  transitions do not bootstrap future value.
- `terminated` and `truncated` follow Gymnasium semantics. Reset and step tuple shapes
  remain compatible with the supported Gymnasium version in `pyproject.toml`.
- Gymnasium 1.x built-in discrete models are accessed explicitly through
  `env.unwrapped.P`; custom wrappers expose their own model through a `P` property.
- Continuous-space bin boundaries, clipping, and edge observations map consistently.

## Supply evidence

- Reproduce bugs through the public wrapper and algorithm path before the fix.
- Add focused regression tests under `tests/` for indexing, probability, reward,
  terminal, and boundary behavior affected by the change.
- Where a model approximates a continuous environment, document what is exact and
  what is an approximation. Do not present a smoke test as model validation.
- Run the focused environment tests and relevant planning or learning integration
  tests. Report slow, stochastic, or unsupported scenarios that were not exercised.
