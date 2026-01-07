"""bettermdptools.experiments.algorithms

Dispatch helpers that map user-facing algorithm names to implementations
in `bettermdptools.algorithms`.

This module keeps the experiment layer lightweight by connecting parameters
to existing implementations without modifying their internals.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .types import EnvBundle


def normalize_algo_name(algo: str) -> str:
    a = algo.strip().lower()
    aliases = {
        "value_iteration": "vi",
        "value-iteration": "vi",
        "vi": "vi",
        "policy_iteration": "pi",
        "policy-iteration": "pi",
        "pi": "pi",
        "q": "q_learning",
        "q-learning": "q_learning",
        "q_learning": "q_learning",
        "sarsa": "sarsa",
    }
    return aliases.get(a, a)


def run_algorithm(algo: str, bundle: EnvBundle, **algo_kwargs: Any) -> Dict[str, Any]:
    """Run a single algorithm and return results in a standardized format.

    Returns
    -------
    dict
        A dictionary containing learned values and policies. When applicable,
        the result includes a policy `pi`.
    """
    a = normalize_algo_name(algo)

    if a in {"vi", "pi"}:
        from bettermdptools.algorithms.planner import Planner

        planner = Planner(bundle.P)
        if a == "vi":
            V, V_track, pi = planner.value_iteration(**algo_kwargs)
            return {"V": V, "V_track": V_track, "pi": pi}
        else:
            V, V_track, pi = planner.policy_iteration(**algo_kwargs)
            return {"V": V, "V_track": V_track, "pi": pi}

    if a in {"q_learning", "sarsa"}:
        from bettermdptools.algorithms.rl import RL

        agent = RL(bundle.env)
        # Pass tabular environment adapters by default while allowing overrides
        algo_kwargs = {
            "nS": bundle.nS,
            "nA": bundle.nA,
            "convert_state_obs": bundle.convert_state_obs,
            **algo_kwargs,
        }
        if a == "q_learning":
            Q, V, pi, Q_track, pi_track, rewards = agent.q_learning(**algo_kwargs)
        else:
            Q, V, pi, Q_track, pi_track, rewards = agent.sarsa(**algo_kwargs)

        return {
            "Q": Q,
            "V": V,
            "pi": pi,
            "Q_track": Q_track,
            "pi_track": pi_track,
            "rewards": rewards,
        }

    raise ValueError(
        f"Unknown algorithm {algo!r}. Supported: vi, pi, q_learning (q), sarsa."
    )
