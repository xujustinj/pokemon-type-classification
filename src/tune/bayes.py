import os

from skopt import BayesSearchCV as SKBayesSearchCV


_CPUS = len(os.sched_getaffinity(0))
# use one fewer core to avoid hogging all resources
_PARALLELISM = max(_CPUS - 1, 1)
print(f"Using {_PARALLELISM} cores")

class BayesSearchCV(SKBayesSearchCV):
    """
    A thin wrapper over skopt's BayesSearchCV class that automatically selects
    parallelism based on the number of cores in the machine.
    """
    def __init__(self, **kwargs):
        super().__init__(n_jobs=_PARALLELISM, **kwargs)
