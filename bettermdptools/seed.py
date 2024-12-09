import os
import random

import numpy as np
from gymnasium.utils import seeding

SEED = None


def set_seed(seed: int) -> None:
    global SEED
    SEED = seed
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    seeding.np_random(SEED)
