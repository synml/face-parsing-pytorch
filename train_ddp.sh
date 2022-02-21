#!/bin/bash

python -m torch.distributed.run --nproc_per_node=8 train.py