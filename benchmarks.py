from contextlib import contextmanager
from joblib import Memory
import hermes.backend.redis
import time
import numpy as np

jcache = Memory('/tmp/foo-cache').cache
hcache = hermes.Hermes(hermes.backend.redis.Backend, ttl=None, port='6380', db=0)

@jcache
def f(n):
    return np.random.randn(n, n)

@hcache
def g(n):
    return np.random.randn(n, n)

@contextmanager
def timer(label):
        start = time.time()
        yield
        end = time.time()
        print('{} - {}s'.format(label, end - start))

with timer("f miss 1000"):
    f(1000)
with timer("f hit 1000"):
    f(1000)

with timer("g miss 1000"):
    g(1000)
with timer("g hit 1000"):
    g(1000)

with timer("f miss 5000"):
    f(5000)
with timer("f hit 5000"):
    f(1000)

with timer("g miss 5000"):
    g(5000)
with timer("g hit 5000"):
    g(5000)

