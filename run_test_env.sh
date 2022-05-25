#!/usr/bin/env bash
  
module add cuda/11.4

source ~/my_envs/envs/bin/activate

module add cuda/11.4

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="./":$PYTHONPATH \
accelerate launch --config_file ./gpu-2.acc  ./test_env.py $*

# PYTHONPATH="./":$PYTHONPATH \
# python ./test_env.py $*

