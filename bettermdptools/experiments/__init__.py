"""High-level experiment workflows.

The primary entrypoint for new users is:

- `run(...)` - run a single algorithm/environment experiment

Optionally:
- `ExperimentBuilder` - fluent configuration API
"""

from .run import run, ExperimentBuilder

__all__ = ["run", "ExperimentBuilder"]
