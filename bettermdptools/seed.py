import os
import random

import numpy as np
from gymnasium.utils import seeding

SEED = None


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set.

    Returns
    -------
    None
    """
    global SEED
    SEED = seed
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    seeding.np_random(SEED)
    print('seed')

def get_seed() -> int:
    """
    Retrieve the current global seed.

    Returns
    -------
    int
        The current seed value set by `set_seed`, or `None` if no seed has been set.
    """
    return SEED