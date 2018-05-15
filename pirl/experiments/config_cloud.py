# Symlink config_local to this file when running in AWS

import socket
import os.path as osp
hostname = socket.gethostname()

SHARED_MNT = '/mnt/efs/population-irl'
DATA_DIR = osp.join(SHARED_MNT, hostname, 'data')
LOG_DIR = osp.join(SHARED_MNT, hostname, 'logs')
CACHE_DIR = osp.join(SHARED_MNT, 'cache')
