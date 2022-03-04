import torch
import math


class CosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, total_iters: int, last_epoch=-1):
        assert total_iters != 0
        self.total_iters = total_iters  # Same as total_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = [base_lr * (1 + math.cos(self.last_epoch * math.pi / self.total_iters)) / 2 for base_lr in self.base_lrs]
        return lr


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, total_iters: int, power=0.9, last_epoch=-1):
        assert total_iters != 0
        self.total_iters = total_iters  # Same as total_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = [base_lr * (1 - self.last_epoch / self.total_iters) ** self.power for base_lr in self.base_lrs]
        return lr
