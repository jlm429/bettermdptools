# Utilities API

The utilities package provides policy evaluation, typed model-free callbacks,
and composable plotting at the existing module paths.

## Callable policy evaluation

TestEnv accepts existing indexable policies and callable policies:

~~~python
import gymnasium as gym

from bettermdptools.utils.test_env import TestEnv

env = gym.make("FrozenLake-v1", is_slippery=False)
try:
    scores = TestEnv.test_env(
        env,
        n_iters=2,
        pi=lambda state, info: 1 if state == 0 else 2,
        seed=7,
    )
finally:
    env.close()
~~~

Supported forms are:

- An indexable mapping or sequence resolved as **policy[state]**
- A callable resolved as **policy(state)**
- A callable resolved as **policy(state, info)**, with positional or
  keyword-only **info**

The callable signature is inspected before invocation. A callable is invoked
once per decision. Exceptions raised by policy code propagate without retry.
TestEnv retains its existing rendering behavior and does not close
caller-owned environments.

## Typed training callbacks

Callbacks can be injected into the RL constructor or into one Q-learning or
SARSA call. A method-level value overrides the constructor value for that run.
An ordered iterable runs each callback in the order supplied.

~~~python
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.callbacks import Callbacks, TransitionContext


class RewardCollector(Callbacks):
    def __init__(self):
        self.rewards = []

    def on_env_step(self, caller, *, context: TransitionContext):
        self.rewards.append(context.reward)


collector = RewardCollector()
Q, V, pi, Q_track, pi_track, rewards = RL(
    env,
    callbacks=collector,
).q_learning(n_episodes=100)
~~~

The public context types are:

- **EpisodeContext**: algorithm name, zero-based episode index, total episode
  count, observation, converted state, current info mapping, episode step
  count, total reward, boundary flags, and the current alpha, epsilon, and
  gamma values.
- **TransitionContext**: the same episode and schedule identity plus the
  one-based transition number, observation, converted state, action, reward,
  termination and truncation flags, next observation, next converted state,
  transition info, and total episode reward including that transition.

**episode_number**, **progress**, and **done** provide convenient derived
values on both context types.

### Lifecycle and ordering

For every episode, hooks run in this order:

1. The environment resets and the observation is converted.
2. **on_episode_begin** receives the start context.
3. **on_episode** receives the same start context.
4. **on_env_step** receives each transition after **env.step** and reward
   accumulation, but before the TD update.
5. Training records the completed episode.
6. **on_episode_end** receives the final episode context.

Context dataclasses are frozen, and their **info** mapping is copied into a
read-only top-level view. Nested observation objects are not copied and should
be treated as read-only. Callback return values are ignored and do not stop or
modify training. Exceptions propagate unchanged, stop the run immediately,
and prevent later callbacks for that hook from running.

### Stable paths and explicit signatures

The **Callbacks** and **MyCallbacks** imports remain available from
**bettermdptools.utils.callbacks**, and the existing hook names remain:

- **on_episode_begin**
- **on_episode**
- **on_env_step**
- **on_episode_end**

Each hook uses the explicit signature **hook(caller, *, context)**. Dispatch
does not inspect callback signatures or adapt older custom method signatures.
It also does not catch callback exceptions. Errors therefore propagate
normally, including TypeError raised inside callback code.

## Composable plotting

Existing Plots methods now accept an optional caller-owned Axes and return the
axes used:

~~~python
import matplotlib.pyplot as plt

from bettermdptools.utils.plots import Plots

fig, ax = plt.subplots()
Plots.values_heat_map(values, "State values", (4, 4), show=False, ax=ax)
fig.tight_layout()
~~~

The rendering helpers always pass an explicit axes to seaborn. They do not
select pyplot's current axes or change global seaborn themes or Matplotlib
rcParams. Omitting **ax** retains the convenient behavior of creating a new
figure.

The directly testable transformations are:

- **Plots.values_to_dataframe(data, size)**
- **Plots.iterations_to_dataframe(data)**
- **Plots.aggregate_values(data, map_size, agg_axes)**
- **Plots.get_policy_map(pi, values, actions, map_size)**

**aggregate_values** accepts numeric measurements only. All axes refer to the
original reshaped array and are reduced simultaneously, so axis ordering is
irrelevant. Empty axes return a copy, negative axes are normalized, and
duplicates are rejected. **get_values_agg_axis_means** remains available with
its existing signature and delegates to these numeric semantics. Policy
labels remain categorical and are never averaged.

Policy maps use an object-backed label array, so multi-character strings such
as "MOVE LEFT" remain intact in NumPy arrays, pandas tables, and rendered
annotations.
