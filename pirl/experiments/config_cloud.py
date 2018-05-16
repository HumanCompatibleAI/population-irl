# Symlink config_local to this file when running in AWS

import socket
import os.path as osp
hostname = socket.gethostname()

SHARED_MNT = '/mnt/efs/population-irl'
LOG_DIR = osp.join(SHARED_MNT, 'logs', hostname)
DATA_DIR = osp.join(SHARED_MNT, 'data')
CACHE_DIR = osp.join(SHARED_MNT, 'cache')
