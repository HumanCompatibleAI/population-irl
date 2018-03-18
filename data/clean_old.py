import itertools
import sys
import os

if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    dates = {}
    for fname in os.listdir(root):
        if fname.endswith('.pkl'):
            fname = fname[:-4]
            components = fname.split('-')
            date = components[-1]
            key = '-'.join(components[:-1])
            dates.setdefault(key, []).append(date)
    dates = {k: sorted(v) for k, v in dates.items()}
    keep = ['{}-{}.pkl'.format(k, v[-1]) for k, v in dates.items()]
    delete = list(itertools.chain(*[['{}-{}.pkl'.format(k, x) for x in v[:-1]] for k, v in dates.items()]))
    print('KEEPING: ', '\n'.join(keep))
    print('DELETING: ', '\n'.join(delete))
    print('Type "DELETE" to confirm')
    if sys.stdin.readline().strip() == 'DELETE':
        for fname in delete:
            os.unlink(os.path.join(root, fname))
