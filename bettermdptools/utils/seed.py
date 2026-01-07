# bettermdptools/utils/seed.py

import os
import random
from typing import Optional

import numpy as np
from gymnasium.utils import seeding


def set_seed(seed: Optional[int]) -> Optional[int]:
    """
    Set global random seeds for reproducibility.

    This function seeds:
    - Python's built-in `random` module
    - NumPy's global RNG
    - Gymnasium's RNG helpers

    Notes
    -----
    Gymnasium does **not** enforce seed usage uniformly across environments.
    Environment-specific behavior may include:
    - Ignoring the provided seed entirely
    - Partially applying the seed
    - Applying the seed only during `env.reset(seed=...)` and not during `env.step(...)`

    As a result, setting a seed improves reproducibility where supported, but does
    **not** guarantee deterministic or bitwise-identical results across runs,
    especially in stochastic or continuous environments.

    Parameters
    ----------
    seed : int or None
        The seed value to set. If None, no global seeding is applied.

    Returns
    -------
    Optional[int]
        The seed used, or None.
    """
    if seed is None:
        return None

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    seeding.np_random(seed)

    return seed
