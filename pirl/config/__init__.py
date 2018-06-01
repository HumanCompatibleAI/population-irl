import logging.config
import numpy as np

from pirl.config import types
from pirl.config.config import RL_ALGORITHMS, SINGLE_IRL_ALGORITHMS, \
        POPULATION_IRL_ALGORITHMS, EXPERIMENTS, LOG_CFG, TENSORFLOW, \
        RAY_SERVER, PROJECT_DIR, EXPERIMENTS_DIR, OBJECT_DIR, CACHE_DIR

def node_setup():
    logging.config.dictConfig(LOG_CFG)
    # Hack: Joblib Memory.cache uses repr() on numpy arrays for metadata.
    # This ends up taking ~100s per call and increases space on disk by 2x.
    # Make numpy repr() more compact.
    np.set_printoptions(threshold=5)
#TODO: Execute this explicitly rather than on import
#It's bad form to mess with the logging settings of scripts importing us.
#Better to set it explicitly in each program during startup.
#However, Ray does not let us run setup code on workers, this is an easy workaround :(
node_setup()

types.validate_config(RL_ALGORITHMS,
                      SINGLE_IRL_ALGORITHMS,
                      POPULATION_IRL_ALGORITHMS)
EXPERIMENTS = {k: types.parse_config(k, v,
                                     RL_ALGORITHMS,
                                     SINGLE_IRL_ALGORITHMS,
                                     POPULATION_IRL_ALGORITHMS)
               for k, v in EXPERIMENTS.items()}