import multiprocessing
import os

def _parallelism() -> int:
    """The level of parallism that this computer can handle.

    Returns:
        parallelism (int): The number of parallel workers that this computer can
            handle.
    """
    try:
        # NOTE: only available on some Unix platforms
        # https://stackoverflow.com/q/74048135
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = multiprocessing.cpu_count()

    return max(cores - 1, 1)

PARALLELISM = _parallelism()
