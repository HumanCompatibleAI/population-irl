import logging.config
import numpy as np

from pirl.config import types
from pirl.config.config import RL_ALGORITHMS, SINGLE_IRL_ALGORITHMS, \
        POPULATION_IRL_ALGORITHMS, EXPERIMENTS, LOG_CFG, TENSORFLOW, \
        RAY_SERVER, PROJECT_DIR, EXPERIMENTS_DIR, OBJECT_DIR, CACHE_DIR

types.validate_config(RL_ALGORITHMS,
                      SINGLE_IRL_ALGORITHMS,
                      POPULATION_IRL_ALGORITHMS)
EXPERIMENTS = {k: types.parse_config(k, v,
                                     RL_ALGORITHMS,
                                     SINGLE_IRL_ALGORITHMS,
                                     POPULATION_IRL_ALGORITHMS)
               for k, v in EXPERIMENTS.items()}