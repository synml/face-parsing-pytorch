#!/bin/bash

chmod u+x train_ddp.sh
python -m torch.distributed.run --nproc_per_node=8 train.py