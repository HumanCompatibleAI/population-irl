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
        print('Note: this does not remove any of the entries by itself!' 
              'More efficient to run redis-cli -p 6380 FLUSHALL.')
        cache.clean()
    else:
        print('Removing cache entries with any of tags', tags)
        cache.clean(tags)

if __name__ == '__main__':
    run()
