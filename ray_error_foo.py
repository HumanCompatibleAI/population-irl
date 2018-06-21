import ray
import time

from mpi4py import MPI

# Without max_calls=1, it hangs forever (but does do some computation).
# With max_calls=1, processes die.
@ray.remote(max_calls=1)
def g(x):
    print('g', x)
    time.sleep(1)
    print('g done', x)
    return x
