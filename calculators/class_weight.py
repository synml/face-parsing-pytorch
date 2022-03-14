import torch
import torch.utils.data
from tqdm import tqdm


def calculate_class_weight(dataloader: torch.utils.data.DataLoader, num_classes: int):
    frequencies = torch.zeros(num_classes)
    for _, target in tqdm(dataloader, desc='Calculate class weight'):
        target = target.to(torch.uint8)
        count_l = torch.bincount(target.flatten(), minlength=num_classes)
        frequencies += count_l

    class_weights = []
    for frequency in frequencies:
        class_weight = 1 / (torch.log(1.02 + (frequency / frequencies.sum())))
        class_weights.append(class_weight)
    return class_weights
