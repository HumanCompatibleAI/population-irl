import sys
from pirl import utils

def run():
    if len(sys.argv) < 2:
        print('usage: <tag1> [tag2 ... tagn]\nuse * as wildcard for all tags.')
        sys.exit(1)
    tags = sys.argv[1:]

    cache = utils.get_hermes()
    if '*' in tags:
        print('Removing all cache entries')
        cache.clean()
    else:
        print('Removing cache entries with any of tags', tags)
        cache.clean(tags)

if __name__ == '__main__':
    run()
