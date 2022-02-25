import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as TF


def inverse_normalize(tensor: Tensor, mean: Tensor, std: Tensor, inplace=False) -> Tensor:
    tensor = TF.normalize(tensor, (-mean / std).tolist(), (1.0 / std).tolist(), inplace)
    return tensor


def inverse_to_tensor_normalize(tensor: Tensor) -> Tensor:
    return tensor.mul_(255).to(torch.uint8)


def draw_segmentation_masks(images: Tensor, masks: Tensor, colors: list, alpha=0.34, gamma=10) -> Tensor:
    assert images.dtype == torch.uint8, f'The images dtype must be uint8, got {images.dtype}'
    assert images.dim() == 4, 'Pass batches, not individual images'
    assert images.size()[1] == 3, 'Pass RGB images. Other Image formats are not supported'
    assert images.shape[-2:] == masks.shape[-2:], 'The images and the masks must have the same height and width'
    assert masks.ndim == 3, 'The masks must be of shape (Batch, H, W)'
    assert masks.dtype == torch.int64, f'The masks must be of dtype int64. Got {masks.dtype}'
    assert images.device == masks.device, 'The device of images and masks must be the same'
    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1. 0 means full transparency, 1 means no transparency'
    assert len(colors[0]) == 3, 'The colors must be RGB format'

    n, h, w = masks.size()
    colored_mask = torch.zeros([n, 3, h, w], dtype=torch.uint8, device=masks.device)
    r = colored_mask[:, 0, :, :]
    g = colored_mask[:, 1, :, :]
    b = colored_mask[:, 2, :, :]
    for i, color in enumerate(colors):
        r[masks == i] = color[0]
        g[masks == i] = color[1]
        b[masks == i] = color[2]

    if alpha == 1:
        return colored_mask
    else:
        alpha_colored_mask = images * (1 - alpha) + colored_mask * alpha + gamma
        alpha_colored_mask = alpha_colored_mask.clamp(0, 255).to(torch.uint8)
        return alpha_colored_mask


def generate_color_palette(num_classes: int) -> list[tuple]:
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_classes)]


# Validate dataset loading code
def show_dataset(images: Tensor, targets: Tensor):
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
