#!/bin/bash

# You can easily run this after chmod with the following command: ./train_ddp.sh
chmod u+x train_ddp.sh

# Set the variable according to the environment of node for optimal performance.
export OMP_NUM_THREADS=128

# Set the variable according to the number of GPUs in the node.
python -m torch.distributed.run --nproc_per_node=8 train.py