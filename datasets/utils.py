from typing import Union

import matplotlib.pyplot as plt
import torch
import torchvision


def draw_segmentation_masks(images: torch.Tensor,
                            masks: torch.Tensor,
                            colors: Union[list, tuple],
                            alpha: float = 0.5,
                            ignore_index: int = None,
                            ignore_color: Union[list, tuple] = None):
    assert images.dtype == torch.float32, f'The images dtype must be float32, got {images.dtype}'
    assert images.dim() == 4, 'Pass batches, not individual images'
    assert images.size()[1] == 3, 'Pass RGB images. Other Image formats are not supported'
    assert images.shape[-2:] == masks.shape[-2:], 'The images and the masks must have the same height and width'
    assert masks.ndim == 3, 'The masks must be of shape (Batch, H, W)'
    assert masks.dtype == torch.int64, f'The masks must be of dtype int64. Got {masks.dtype}'
    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1. 0 means full transparency, 1 means no transparency'

    # 각 채널 별로 디코딩하기 위해 복사
    r = masks.clone()
    g = masks.clone()
    b = masks.clone()

    # Assign colors according to class for each channel (각 채널 별로 class에 따라 색상 대입)
    for i in range(len(colors)):
        r[masks == i] = colors[i][0]
        g[masks == i] = colors[i][1]
        b[masks == i] = colors[i][2]
    if ignore_index and ignore_color is not None:
        r[masks == ignore_index] = ignore_color[0]
        g[masks == ignore_index] = ignore_color[1]
        b[masks == ignore_index] = ignore_color[2]

    decoded_masks = (r.unsqueeze(dim=1), g.unsqueeze(dim=1), b.unsqueeze(dim=1))
    decoded_masks = torch.cat(decoded_masks, dim=1).to(torch.float32)
    decoded_masks /= 255

    if alpha == 1:
        return decoded_masks
    else:
        alpha_decoded_mask = images * (1 - alpha) + decoded_masks * alpha
        return alpha_decoded_mask


# Validate dataset loading code
def show_dataset(images: torch.Tensor, targets: torch.Tensor):
    to_pil_image = torchvision.transforms.ToPILImage()
    plt.rcParams['figure.figsize'] = (17, 6)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False

    assert images.shape[0] == targets.shape[0]
    for i in range(images.shape[0]):
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title('Input image')
        axs[0].imshow(to_pil_image(images[i].cpu()))
        axs[1].set_title('Groundtruth')
        axs[1].imshow(targets[i].cpu())
        plt.show()
