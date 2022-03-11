import torch
from tqdm import tqdm


def calculate_class_weight(dataloader, num_classes):
    z = torch.zeros(num_classes)
    for sample in tqdm(dataloader, desc='Calculate class weight'):
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(torch.uint8)
        count_l = torch.bincount(labels, minlength=num_classes)
        z += count_l
    total_frequency = torch.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (torch.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    return class_weights
