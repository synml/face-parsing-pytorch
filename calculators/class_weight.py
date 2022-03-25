import torch
import torch.utils.data
from tqdm import tqdm


def calculate_class_weight(dataloader: torch.utils.data.DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for _, target in tqdm(dataloader, desc='Calculate class weight'):
        target = target.to(torch.uint8)
        count = torch.bincount(target.flatten(), minlength=num_classes)
        counts += count

    weight = torch.median(counts) / counts

    with open('class_weight.txt', 'w', encoding='utf-8') as f:
        f.write(weight.tolist())
    return weight
