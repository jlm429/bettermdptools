"""High-level experiment workflows.

The primary entrypoint for new users is:

- `run(...)` - run a single algorithm/environment experiment

Optionally:
- `ExperimentBuilder` - fluent configuration API
"""

from .run import run, ExperimentBuilder

__all__ = ["run", "ExperimentBuilder"]

# Optional Optuna add-on
try:
    from .optuna import optimize, OptunaResult, MissingOptunaDependency

    __all__ += ["optimize", "OptunaResult", "MissingOptunaDependency"]
except Exception:
    # Optuna is optional; do not fail base imports
    pass
