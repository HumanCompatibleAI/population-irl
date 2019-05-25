# population-irl

Code accompanying the paper [Multi-task Maximum Entropy Inverse Reinforcement Learning](https://arxiv.org/abs/1805.08882).

## Installation

I use Anaconda for dependencies, you can create an environment with the prerequisite dependencies by running:
	
	conda env create name pirl
	source activate pirl
	conda env update --name pirl --file environment.yml 

There are some binary libraries that some Python packages required that are not in Conda. On Ubuntu you can install the necessary dependencies by:
  sudo apt install zlib1g-dev libopenmpi-dev 

Alternately, if you wish to configure your own container, the dependencies are RLLab, Python 3, NumPy, PyTorch and OpenAI Gym.

## Troubleshooting

This codebase sometimes triggers a double-free error, which seems to be due to an interaction between jemalloc and TensorFlow, see e.g. https://github.com/tensorflow/tensorflow/issues/6968 If you run into this issue, you can workaround it by using tcmalloc:

	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:$LD_PRELOAD

## Usage

To check everything works, you can run:

	redis-server ./config/redis/default.conf
	python run_experiments.py dummy-test few-dummy-test dummy-continuous-test
	pytest

If you need to restrict to run on a subset of GPUs, use CUDA_VISIBLE_DEVICES. Since Ray does not support fractional resources, we pretend to have more GPUs than we actually do by a factor of run_experiments.GPU_MULTIPLIER. You'll need to repeat the GPU ID in CUDA_VISIBLE_DEVICES by this multipler for things to work.

## Jupyter notebooks

There are some Jupyter notebooks in analysis/ that are used for ad-hoc tests/figure generation. Note since these are in a subdirectory, you'll need to set an appropriate PYTHONPATH (which should be an absolute, not relative, path).
