import os

import torch
import torchvision
import tqdm

import datasets
import utils

if __name__ == '__main__':
    # Create components builder
    builder = utils.builder.Builder()
    config = builder.config
    model_name = builder.model_name
    amp_enabled = config['train']['amp_enabled']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Dataset
    valset, valloader = builder.build_dataset('test')

    # 2. Model
    model = builder.build_model(valset.num_classes, pretrained=True).to(device)
    model.eval()
    print(f'Activated model: {model_name}')

    # Collect image names
    image_names = [os.path.basename(image_path) for image_path in valset.images]

    # Save segmentation results
    result_dir = os.path.join('demo', model_name.lower())
    groundtruth_dir = os.path.join('demo', 'groundtruth')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    for i, (images, targets) in enumerate(tqdm.tqdm(valloader, desc='Demo')):
        images, targets = images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)

        mean = torch.tensor(valset.transforms.normalize.mean)
        std = torch.tensor(valset.transforms.normalize.std)
        images = datasets.utils.inverse_to_tensor_normalize(datasets.utils.inverse_normalize(images, mean, std))
        outputs = datasets.utils.draw_segmentation_masks(images, outputs, valset.colors)
        targets = datasets.utils.draw_segmentation_masks(images, targets, valset.colors)

        # process per 1 batch
        for j, (output, target) in enumerate(zip(outputs, targets, strict=True)):
            file_name = image_names[targets.shape[0] * i + j]
            torchvision.io.write_jpeg(output.cpu(), os.path.join(result_dir, file_name), quality=100)
            torchvision.io.write_jpeg(target.cpu(), os.path.join(groundtruth_dir, file_name), quality=100)
