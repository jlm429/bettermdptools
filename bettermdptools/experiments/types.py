"""bettermdptools.experiments.types

Lightweight types used by experiment entry points.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class EnvBundle:
    """A compact adapter bundle produced by EnvFactory.

    Attributes
    ----------
    env:
        A gymnasium environment instance, which may be wrapped.
    P:
        Transition dictionary in Gym-style discrete format.
    convert_state_obs:
        Function that maps an environment observation to an integer state index.
        For most discrete Gym environments, this is the identity function.
    nS, nA:
        Number of states and actions for tabular methods.
    meta:
        Optional metadata such as wrapper name or environment details.
    """

    env: Any
    P: Dict
    convert_state_obs: Callable[[Any], int]
    nS: int
    nA: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Results returned by the `run(...)` entry point.

    The `train` field contains algorithm-specific outputs.
    The `eval` field is optional and contains evaluation results such as
    test episode returns.
    """

    algo: str
    env_id: str
    seed: Optional[int]
    train: Dict[str, Any]
    eval: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
