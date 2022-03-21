from collections import namedtuple
import glob
import json
import os
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import torchvision.datasets.utils
import torchvision.transforms.functional as TF
import tqdm


class Lane(torchvision.datasets.VisionDataset):
    LaneClass = namedtuple('LaneClass', ['name', 'id', 'color'])
    classes = [
        LaneClass('background', 0, (0, 0, 0)),
        LaneClass('white_lane', 1, (255, 255, 255)),
        LaneClass('yellow_lane', 2, (255, 128, 0)),
        LaneClass('stop_line', 3, (255, 255, 255)),
    ]

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super(Lane, self).__init__(root, transforms, transform, target_transform)
        assert split in ('train', 'val')
        self.split = split
        self.colors = [cls.color for cls in self.classes]
        self.num_classes = len(self.classes)
        self.images = []
        self.targets = []

        if not (os.path.exists(os.path.join(self.root, 'Training', 'preprocessed_mask')) and
                len(os.listdir(os.path.join(self.root, 'Training', 'preprocessed_mask'))) == 30000):
            self.preprocess()

        if self.split == 'train':
            self.images = glob.glob(os.path.join(self.root, 'Training', 'image', '*'))
            self.targets = glob.glob(os.path.join(self.root, 'Training', 'preprocessed_mask', '*'))
        else:
            self.images = glob.glob(os.path.join(self.root, 'Validation', 'image', '*'))
            self.targets = glob.glob(os.path.join(self.root, 'Validation', 'preprocessed_mask', '*'))

        try:
            self.images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.targets.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            self.images.sort()
            self.targets.sort()

    def preprocess(self):
        image_paths = glob.glob(os.path.join(self.root, 'Training', 'image', '*')) + \
                      glob.glob(os.path.join(self.root, 'Validation', 'image', '*'))
        gt_paths = glob.glob(os.path.join(self.root, 'Training', 'gt', '*')) + \
                   glob.glob(os.path.join(self.root, 'Validation', 'gt', '*'))
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        gt_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        for (image_path, gt_path) in tqdm.tqdm(zip(image_paths, gt_paths, strict=True), 'Preprocess dataset',
                                               total=len(gt_paths)):
            with open(gt_path) as f:
                gt = json.load(f)
            image = cv2.imread(image_path)
            mask = np.zeros_like(image)

            for i in gt['annotations']:
                # get point
                points = []
                for j in i['data']:
                    points.append([j['x'], j['y']])
                points = np.array(points)

                # check class [blue,green,red]
                if i['class'] == 'traffic_lane':
                    if i['attributes'][1]['value'] == 'dotted':
                        continue
                    if i['attributes'][0]['value'] == 'white':
                        mask = cv2.polylines(mask, [points], False, (1, 1, 1), 3)
                    elif i['attributes'][0]['value'] == 'yellow':
                        mask = cv2.polylines(mask, [points], False, (2, 2, 2), 3)
                elif i['class'] == "stop_line":
                    mask = cv2.polylines(mask, [points], False, (3, 3, 3), 3)

            os.makedirs(os.path.dirname(image_path).replace('image', 'preprocessed_mask'), exist_ok=True)
            cv2.imwrite(image_path.replace('image', 'preprocessed_mask').replace('.jpg', '.png'), mask)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torchvision.io.read_image(self.images[index], torchvision.io.ImageReadMode.RGB)
        target = torchvision.io.read_image(self.targets[index], torchvision.io.ImageReadMode.GRAY)

        image = TF.resize(image, [1280, 720], TF.InterpolationMode.BILINEAR, antialias=True)
        target = TF.resize(target, [1280, 720], TF.InterpolationMode.BILINEAR, antialias=True)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ['Split: {split}', 'Type: {target_type}']
        return '\n'.join(lines).format(**self.__dict__)


if __name__ == '__main__':
    dataset = Lane('../../../data/Lane', 'train')
    exit()
