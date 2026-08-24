---
name: algorithm-changes
description: Change planning or model-free reinforcement-learning algorithms in bettermdptools with focused behavioral evidence and API compatibility. Use for work in bettermdptools/algorithms, not adapter-only or documentation-only tasks.
---

# Change an algorithm

Read the repository [agent guide](../../../AGENTS.md) first.

## Establish the contract

1. Inspect the relevant implementation in `bettermdptools/algorithms/`, its public
   call sites in `tests/test_envs.py` and `examples/`, and affected utility callbacks.
2. For a bug, reproduce the failure through `Planner` or `RL` before changing code.
3. Identify convergence, update, exploration, discounting, terminal-state, dtype,
   shape, tracking, and randomization behavior affected by the change.

## Implement narrowly

- Preserve method signatures, defaults, return ordering, array shapes, and policy
  representation unless a breaking change is explicit.
- For planning, treat each transition as `(probability, next_state, reward, done)`.
  Confirm probability weighting and prevent bootstrapping beyond terminal states.
- For Q-learning or SARSA, preserve Gymnasium reset and step semantics and examine
  termination and truncation separately before choosing a target update.
- Avoid coupling algorithm fixes to environment-specific assumptions.

## Supply evidence

- Add focused coverage in `tests/` for the observable failure or behavior. Include
  small deterministic models when they make expected values and policies exact.
- Compare scalar and vectorized planning results when changing shared planning
  semantics.
- Run the focused tests, then broader existing tests when practical. Report seeds,
  warnings, nondeterministic limits, and untested paths.
- Review the final diff for public API drift, unrelated refactors, weakened tests,
  dependency changes, and generated notebook output.
