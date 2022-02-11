from collections import namedtuple
import glob
import os
from typing import Callable, List, Optional, Union, Tuple
import zipfile

import cv2
import numpy as np
from PIL import Image
import torchvision
import torchvision.datasets.utils
import tqdm


class CelebAMaskHQ(torchvision.datasets.VisionDataset):
    """CelebAMask-HQ Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html>

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly, dataset is selected.
        target_type (string or list, optional): Type of target to use, ``mask``, ``pose``, or ``attr``.
            Can also be a list to output a tuple with all specified target types.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
    """

    CelebAMaskHQClass = namedtuple('CelebAMaskHQClass', ['name', 'id'])
    classes = [
        CelebAMaskHQClass('background', 0),
        CelebAMaskHQClass('skin', 1),
        CelebAMaskHQClass('l_brow', 2),
        CelebAMaskHQClass('r_brow', 3),
        CelebAMaskHQClass('l_eye', 4),
        CelebAMaskHQClass('r_eye', 5),
        CelebAMaskHQClass('eye_g', 6),
        CelebAMaskHQClass('l_ear', 7),
        CelebAMaskHQClass('r_ear', 8),
        CelebAMaskHQClass('ear_r', 9),
        CelebAMaskHQClass('nose', 10),
        CelebAMaskHQClass('mouth', 11),
        CelebAMaskHQClass('u_lip', 12),
        CelebAMaskHQClass('l_lip', 13),
        CelebAMaskHQClass('neck', 14),
        CelebAMaskHQClass('neck_l', 15),
        CelebAMaskHQClass('cloth', 16),
        CelebAMaskHQClass('hair', 17),
        CelebAMaskHQClass('hat', 18),
    ]

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 target_type: Union[List[str], str] = 'mask',
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super(CelebAMaskHQ, self).__init__(root, transforms, transform, target_transform)
        assert split in ('train', 'valid', 'test', 'all')
        assert target_type in ('mask', 'pose', 'attr')
        self.split = split
        self.target_type = target_type
        self.images = []
        self.targets = []
        self.class_names = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear',
                            'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.num_classes = 19

        if download:
            self.download()

        if not os.path.exists(os.path.join(self.root, 'preprocessed_mask')):
            self.preprocess()

        if self.split == 'all':
            self.images = glob.glob(os.path.join(self.root, 'CelebA-HQ-img', '*'))
            self.targets = glob.glob(os.path.join(self.root, 'preprocessed_mask', '*'))
        else:
            # Load mapping information
            orig_to_hq_mapping = {}
            for s in open(os.path.join(self.root, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r'):
                if s.startswith('idx'):
                    continue
                idx, _, orig_file = s.split()
                orig_to_hq_mapping[orig_file] = idx

            # Load split list
            for s in open(os.path.join(self.root, 'list_eval_partition.txt'), 'r'):
                orig_file, split_idx = s.split()
                if orig_file not in orig_to_hq_mapping:
                    continue
                hq_id = orig_to_hq_mapping[orig_file]
                if self.split == 'train' and split_idx == '0':
                    self.images.append(os.path.join(self.root, 'CelebA-HQ-img', hq_id + '.jpg'))
                elif self.split == 'valid' and split_idx == '1':
                    self.images.append(os.path.join(self.root, 'CelebA-HQ-img', hq_id + '.jpg'))
                elif self.split == 'test' and split_idx == '2':
                    self.images.append(os.path.join(self.root, 'CelebA-HQ-img', hq_id + '.jpg'))
            self.targets = [i.replace('CelebA-HQ-img', 'preprocessed_mask').replace('.jpg', '.png')
                            for i in self.images]

        self.images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.targets.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def download(self):
        dataset_file = {'id': '1PtttcVHOjC5-9xSBPWNHh0OOUrEgHWcl',
                        'dir': os.path.dirname(self.root),
                        'name': 'CelebAMask-HQ.zip',
                        'md5': 'f1d85e89ae6ac8c1cee9f1278095ce09'}
        dataset_file_path = os.path.join(dataset_file['dir'], dataset_file['name'])

        if torchvision.datasets.utils.check_integrity(dataset_file_path, dataset_file['md5']):
            print('A file already downloaded and verified.')
            return

        print('Download a dataset archive. . .')
        torchvision.datasets.utils.download_file_from_google_drive(*dataset_file.values())

        print('Extract the dataset archive. . .')
        with zipfile.ZipFile(dataset_file_path, 'r') as f:
            f.extractall(dataset_file['dir'])
        print('Extraction complete.')

    def preprocess(self):
        orig_mask_path = os.path.join(self.root, 'CelebAMask-HQ-mask-anno')
        mask_path = os.path.join(self.root, 'preprocessed_mask')
        os.makedirs(mask_path, exist_ok=True)

        # i는 orig_mask_path의 하위 폴더 0 ~ 14를 지정
        # j는 각 하위 폴더의 이미지 index 범위를 지정
        for i in tqdm.tqdm(range(15), desc='Preprocess dataset'):
            for j in tqdm.tqdm(range(i * 2000, (i + 1) * 2000), leave=False):
                mask = np.zeros((512, 512))
                for cls in self.classes[1:]:
                    file_name = str(j).zfill(5) + '_' + cls.name + '.png'
                    path = os.path.join(orig_mask_path, str(i), file_name)
                    if os.path.exists(path):
                        orig_mask = np.array(Image.open(path).convert('L'))
                        mask[orig_mask == 255] = cls.id
                cv2.imwrite(os.path.join(mask_path, f'{j}.png'), mask)

    def __getitem__(self, index) -> Tuple[Image.Image, Image.Image]:
        image = Image.open(self.images[index]).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)

        target = Image.open(self.targets[index]).convert('L')

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ['Split: {split}', 'Type: {target_type}']
        return '\n'.join(lines).format(**self.__dict__)
