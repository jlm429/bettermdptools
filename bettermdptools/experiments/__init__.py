"""High-level experiment and hyperparameter-search workflows.

`run` executes one algorithm and environment combination and returns a
`RunResult`. `ExperimentBuilder` provides the same workflow through a fluent
interface. `optimize` adds optional Optuna-based hyperparameter search and only
imports Optuna when a search starts.
"""

from .optuna import MissingOptunaDependency, OptunaResult, optimize
from .run import ExperimentBuilder, run
from .types import RunResult

__all__ = [
    "ExperimentBuilder",
    "MissingOptunaDependency",
    "OptunaResult",
    "RunResult",
    "optimize",
    "run",
]
