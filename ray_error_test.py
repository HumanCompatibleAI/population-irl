import ray
import time

import ray_error_foo

@ray.remote
def f(x):
    print('f', x)
    res = ray.get([ray_error_foo.g.remote(y) for y in range(x)])
    print(res)
    time.sleep(1)
    print('f done', x)
    return sum(res)

if __name__ == '__main__':
    ray.init(num_cpus=4, num_gpus=1, redirect_worker_output=True)

    print(ray.get([f.remote(i) for i in range(4)]))
