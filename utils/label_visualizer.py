import argparse
import glob
import os
import torch
import torchvision
import tqdm
from collections import namedtuple
from torch import Tensor

CelebAMaskHQClass = namedtuple('CelebAMaskHQClass', ['name', 'id', 'color'])
classes = [
    CelebAMaskHQClass('background', 0, (0, 0, 0)),
    CelebAMaskHQClass('skin', 1, (204, 0, 0)),
    CelebAMaskHQClass('l_brow', 2, (0, 255, 255)),
    CelebAMaskHQClass('r_brow', 3, (255, 204, 204)),
    CelebAMaskHQClass('l_eye', 4, (51, 51, 255)),
    CelebAMaskHQClass('r_eye', 5, (204, 0, 204)),
    CelebAMaskHQClass('eye_g', 6, (204, 204, 0)),
    CelebAMaskHQClass('l_ear', 7, (102, 51, 0)),
    CelebAMaskHQClass('r_ear', 8, (255, 0, 0)),
    CelebAMaskHQClass('ear_r', 9, (0, 204, 204)),
    CelebAMaskHQClass('nose', 10, (76, 153, 0)),
    CelebAMaskHQClass('mouth', 11, (102, 204, 0)),
    CelebAMaskHQClass('u_lip', 12, (255, 255, 0)),
    CelebAMaskHQClass('l_lip', 13, (0, 0, 153)),
    CelebAMaskHQClass('neck', 14, (255, 153, 51)),
    CelebAMaskHQClass('neck_l', 15, (0, 51, 0)),
    CelebAMaskHQClass('cloth', 16, (0, 204, 0)),
    CelebAMaskHQClass('hair', 17, (0, 0, 204)),
    CelebAMaskHQClass('hat', 18, (255, 51, 153)),
]
colors = [cls.color for cls in classes]


def draw_segmentation_mask(image: Tensor, mask: Tensor, colors: list, alpha=0.4, gamma=20) -> Tensor:
    assert image.dtype == torch.uint8, f'The images dtype must be uint8, got {image.dtype}'
    assert image.dim() == 3, 'The images must be of shape (C, H, W)'
    assert image.size()[0] == 3, 'Pass RGB images. Other Image formats are not supported'
    assert image.shape[-2:] == mask.shape[-2:], 'The images and the masks must have the same height and width'
    assert mask.ndim == 2, 'The masks must be of shape (H, W)'
    assert mask.dtype == torch.uint8, f'The masks must be of dtype uint8. Got {mask.dtype}'
    assert image.device == mask.device, 'The device of images and masks must be the same'
    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1. 0 means full transparency, 1 means no transparency'
    assert len(colors[0]) == 3, 'The colors must be RGB format'

    h, w = mask.size()
    colored_mask = torch.zeros((3, h, w), dtype=torch.uint8, device=mask.device)
    r = colored_mask[ 0, :, :]
    g = colored_mask[1, :, :]
    b = colored_mask[2, :, :]
    for i, color in enumerate(colors):
        r[mask == i] = color[0]
        g[mask == i] = color[1]
        b[mask == i] = color[2]

    if alpha == 1:
        return colored_mask
    else:
        alpha_colored_mask = image * (1 - alpha) + colored_mask * alpha + gamma
        alpha_colored_mask = alpha_colored_mask.clamp(0, 255).to(torch.uint8)
        return alpha_colored_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='src', help='Directory where source data is stored')
    parser.add_argument('--dest', type=str, default='results', help='directory to store results')
    parser.add_argument('--device', type=str, default='auto', help='device to use (auto, cpu, cuda)')
    args = parser.parse_args()

    # Device 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # source 폴더에서 데이터 경로 로드
    assert os.path.exists(args.src), 'src is not exists'
    assert os.path.isdir(args.src), 'src must be a directory'
    labels = glob.glob(os.path.join(args.src, '*.grayscale.png'))
    images = [i.replace('.grayscale.png', '.png') for i in labels]
    assert len(images) > 0 and len(labels) > 0, 'No file in src'
    assert len(images) == len(labels)

    # 정렬
    try:
        images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        labels.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        images.sort()
        labels.sort()

    os.makedirs(args.dest, exist_ok=True)
    for image_path, label_path in tqdm.tqdm(zip(images, labels), total=len(images)):
        # 이미지 로드
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB).to(device)
        label = torchvision.io.read_image(label_path, torchvision.io.ImageReadMode.GRAY).to(device)
        label.squeeze_()

        # 라벨 데이터로 색칠
        colored_label = draw_segmentation_mask(image, label, colors)

        # 색칠한 결과 저장
        file_name = os.path.basename(label_path)
        torchvision.io.write_png(colored_label.cpu(), os.path.join(args.dest, file_name))
