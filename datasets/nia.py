import glob
import os
from collections import namedtuple
from typing import Callable, Optional, Tuple

import torch
import torchvision
import torchvision.datasets.utils


class NIA(torchvision.datasets.VisionDataset):
    """CelebAMask-HQ Dataset

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'test'}.
            Accordingly, dataset is selected.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
    """

    NIAClass = namedtuple('NIAClass', ['name', 'id', 'color'])
    classes = [
        NIAClass('background', 0, (0, 0, 0)),
        NIAClass('skin', 1, (204, 0, 0)),
        NIAClass('l_brow', 2, (0, 255, 255)),
        NIAClass('r_brow', 3, (255, 204, 204)),
        NIAClass('l_eye', 4, (51, 51, 255)),
        NIAClass('r_eye', 5, (204, 0, 204)),
        NIAClass('eye_g', 6, (204, 204, 0)),
        NIAClass('l_ear', 7, (102, 51, 0)),
        NIAClass('r_ear', 8, (255, 0, 0)),
        NIAClass('ear_r', 9, (0, 204, 204)),
        NIAClass('nose', 10, (76, 153, 0)),
        NIAClass('mouth', 11, (102, 204, 0)),
        NIAClass('u_lip', 12, (255, 255, 0)),
        NIAClass('l_lip', 13, (0, 0, 153)),
        NIAClass('neck', 14, (255, 153, 51)),
        NIAClass('neck_l', 15, (0, 51, 0)),
        NIAClass('cloth', 16, (0, 204, 0)),
        NIAClass('hair', 17, (0, 0, 204)),
        NIAClass('hat', 18, (255, 51, 153)),
    ]

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super(NIA, self).__init__(root, transforms, transform, target_transform)
        assert split in ('train', 'test')
        self.split = split
        self.colors = [cls.color for cls in self.classes]
        self.num_classes = len(self.classes)
        self.images = glob.glob(os.path.join(self.root, self.split, 'images', '*'))
        self.targets = glob.glob(os.path.join(self.root, self.split, 'labels', '*'))
        self.images.sort()
        self.targets.sort()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torchvision.io.read_image(self.images[index], torchvision.io.ImageReadMode.RGB)
        target = torchvision.io.read_image(self.targets[index], torchvision.io.ImageReadMode.GRAY)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    dataset = NIA('../../../data/NIA', 'train')
    exit()
