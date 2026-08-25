---
name: environment-model-validation
description: Validate changes to bettermdptools environment models, Gymnasium wrappers, adapters, or discretization with explicit transition, reward, and boundary evidence. Use for work in bettermdptools/envs or failures caused by model assumptions.
---

# Validate an environment model or adapter

Read the repository [agent guide](../../../AGENTS.md),
[Gymnasium compatibility](../gymnasium-compatibility/SKILL.md), and
[Testing and validation](../testing-validation/SKILL.md) first.

## Establish actual support

Inspect the affected files under `bettermdptools/envs/`, their construction path in
`bettermdptools/experiments/env_factory.py`, relevant cases in
`tests/test_envs.py` and `tests/test_gymnasium_compat.py`, and the matching example.
Do not infer support from an environment name or claim that every supported
Gymnasium environment has a custom model.

Known custom paths include Blackjack's exact generated transition model and
context-aware observation transform, plus generated discretized models for CartPole
and Acrobot. Other wrapper and model files exist. Inspect their current
implementation, test coverage, construction cost, and assumptions before making a
support claim.

## Trace the representation

1. Trace representative observations from `reset()` and `step()` through the public
   wrapper into the integer state consumed by `Planner` or `RL`.
2. Trace action indices in the reverse direction when a wrapper maps discrete
   actions into an underlying continuous action space.
3. Identify whether `P` comes from `env.unwrapped.P`, a wrapper property, a checked-in
   model, or a generated approximation. Treat wrapped and unwrapped attributes
   deliberately.

## Validate invariants

- State keys and next-state indices are integers in `0..nS-1`. Verify that distinct
  represented states do not collide unintentionally. `Planner` indexes states by
  `range(len(P))`, so sparse or nonzero-based keys are not interchangeable.
- Action keys match `0..nA-1`, and every modeled state has the expected actions.
- Every `(state, action)` has at least one transition. Probabilities are finite,
  lie in `[0, 1]`, and sum to one within a justified numerical tolerance.
- Each planning transition remains `(probability, next_state, reward, done)`.
  Rewards match environment behavior, and `done` prevents bootstrapping only for a
  true model terminal state.
- Wrapper interactions preserve separate `terminated` and `truncated` values.
  Time-limit truncation is an episode boundary, not automatically a terminal state
  in the planning model.
- Discretization covers lower and upper boundaries, clips or rejects out-of-range
  observations deliberately, maps edge values consistently, and declares spaces
  that contain emitted observations and accepted actions.

## Supply behavioral evidence

- Reproduce a model bug through the public wrapper and algorithm path before fixing
  it, then add a focused regression for the original collision, reward, terminal,
  normalization, or boundary failure.
- Check the full model mechanically when its size permits. For large models, use a
  justified sample that includes boundaries, terminal states, and every action.
- Compare representative one-step model outcomes and rewards with actual environment
  steps when practical. For stochastic behavior, use repeated empirical checks and
  a stated tolerance rather than treating one rollout as proof.
- State what is exact and what is approximated. A construction smoke test alone is
  not model validation.
- Run the focused environment and Gymnasium compatibility tests before the full
  suite. Report expensive, stochastic, skipped, or unsupported paths.
