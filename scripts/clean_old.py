import itertools
import shutil
import sys
import os

if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    dates = {}
    for fname in os.listdir(root):
        components = fname.split('-')
        if len(components) < 3:
            continue
        date = components[-2]
        githash = components[-1]
        experiment = '-'.join(components[:-2])
        key = (experiment, githash)
        dates.setdefault(key, []).append(date)
    dates = {k: sorted(v) for k, v in dates.items()}
    keep = ['{}-{}-{}'.format(k[0], v[-1], k[1]) for k, v in dates.items()]
    delete = list(itertools.chain(*[['{}-{}-{}'.format(k[0], x, k[1]) for x in v[:-1]] for k, v in dates.items()]))
    print('KEEPING: ', '\n'.join(keep))
    print('DELETING: ', '\n'.join(delete))
    print('Type "DELETE" to confirm')
    if sys.stdin.readline().strip() == 'DELETE':
        for fname in delete:
            shutil.rmtree(os.path.join(root, fname))