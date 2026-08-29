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

- Confirm the supported Gymnasium version range in `pyproject.toml`; do not copy it
  into another configuration owner.
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

Treat the environment IDs and supported wrapper paths in
`tests/test_gymnasium_compat.py` and `EnvFactory._registry` as authoritative. Do not
copy older IDs from external examples.

When changing `TestEnv` rendering, preserve the complete supplied wrapper stack,
state conversion, seeding, and caller ownership. Run
`tests/test_runtime_correctness.py -k rendering`, which owns modeled-wrapper
coverage, seeded behavior, rendering prerequisites, and cleanup on failures.

## Seed only what the task owns

- For environment reproducibility, pass a seed to `env.reset(seed=...)`.
- Seed `env.action_space` separately when reproducible sampled actions matter.
- A repeated reset seed should reproduce reset observations where the environment
  supports that guarantee. Test it through wrappers, not only on an unwrapped env.
- `bettermdptools.utils.seed.set_seed` seeds global generators on a best-effort
  basis. It does not replace seeding environment resets.
- The experiment entrypoint seeds global generators and the first training and
  evaluation resets. Later episode resets continue each environment's seeded random
  sequence instead of restarting it.

## Evidence

Add focused coverage to `tests/test_gymnasium_compat.py` for API contracts that
change. Reuse small deterministic environments when termination and truncation
targets need exact expected values. Run that focused file before broader environment
tests and the full suite.
