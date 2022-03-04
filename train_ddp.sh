#!/bin/bash

chmod u+x train_ddp.sh
export OMP_NUM_THREADS=1
python -m torch.distributed.run --nproc_per_node=8 train.py