import glob
import os
from typing import Callable, Optional, Tuple

import torch
import torchvision


class DemoDataset(torchvision.datasets.VisionDataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super().__init__(root, transforms, transform, target_transform)
        self.images = glob.glob(os.path.join(self.root, '*'))

        try:
            self.images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            self.images.sort()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None]:
        image = torchvision.io.read_image(self.images[index], torchvision.io.ImageReadMode.RGB)
        target = torch.zeros_like(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = DemoDataset('../demo/input_image')
