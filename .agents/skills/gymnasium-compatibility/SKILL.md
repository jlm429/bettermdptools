---
name: gymnasium-compatibility
description: Change or validate bettermdptools integration with the supported Gymnasium API, including wrappers, reset and step contracts, episode boundaries, model access, environment IDs, and seeding. Use for Gymnasium compatibility work across envs, algorithms, experiments, utilities, tests, or examples.
---

# Maintain Gymnasium compatibility

Read the repository [agent guide](../../../AGENTS.md) and
[Testing and validation](../testing-validation/SKILL.md) first. Read
[Environment model validation](../environment-model-validation/SKILL.md) when a
model, wrapper, adapter, or discretizer changes.

## Use the current contract

- Confirm the supported version range in both `pyproject.toml` and `setup.py`.
  The current range is Gymnasium `>=1.3.0,<1.4`.
- `reset()` returns `(observation, info)`. Wrappers must forward supported reset
  arguments such as `seed` and `options`, preserve the info mapping, and emit an
  observation contained by their declared observation space.
- `step()` returns `(observation, reward, terminated, truncated, info)`. Preserve
  both flags and the info mapping through wrappers and utilities.
- Stop an episode when `terminated or truncated`. For TD targets, bootstrap across
  truncation but not true termination unless an explicitly changed algorithm
  contract requires different behavior.
- Gymnasium built-in discrete transition dictionaries are read from
  `env.unwrapped.P`. bettermdptools wrappers that own a model expose it through
  their public `P` property. Do not rely on wrapper attribute forwarding.

## Inspect every integration point affected

Compatibility can cross `bettermdptools/envs/`, `algorithms/rl.py`,
`experiments/env_factory.py`, `experiments/run.py`, `utils/test_env.py`, tests,
README examples, and notebooks. Trace the normal public path instead of updating a
single tuple unpack in isolation.

Validate transformed observation and action spaces with `space.contains`. Exercise
a short `TimeLimit` path to prove truncation survives wrapper layers. Close every
environment created by a test or diagnostic.

Current compatibility coverage uses `FrozenLake-v1`, `FrozenLake8x8-v1`,
`Taxi-v4`, `Blackjack-v1`, `CartPole-v1`, `Acrobot-v1`, and `Pendulum-v1`. Confirm
IDs against current tests and Gymnasium before adding or replacing one. Do not copy
older IDs from external examples.

## Seed only what the task owns

- For environment reproducibility, pass a seed to `env.reset(seed=...)`.
- Seed `env.action_space` separately when reproducible sampled actions matter.
- A repeated reset seed should reproduce reset observations where the environment
  supports that guarantee. Test it through wrappers, not only on an unwrapped env.
- `bettermdptools.utils.seed.set_seed` seeds global generators on a best-effort
  basis. It does not replace seeding environment resets.
- The current experiment entrypoint reports best-effort global seeding and does not
  establish end-to-end reset seeding. Do not promise stronger reproducibility or
  repair deferred seeding behavior unless seeding is the explicit task.

## Evidence

Add focused coverage to `tests/test_gymnasium_compat.py` for API contracts that
change. Reuse small deterministic environments when termination and truncation
targets need exact expected values. Run that focused file before broader environment
tests and the full suite.
