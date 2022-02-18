import os

import torch.utils.data
import torchvision
import torchvision.transforms.functional as TF
import tqdm

import datasets
import utils

if __name__ == '__main__':
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg, device)

    # 1. Dataset
    valset, valloader = builder.build_dataset('val')

    # 2. Model
    model = builder.build_model(valset.num_classes, pretrained=True)
    model.eval()
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # Collect image names
    image_names = []
    for image_path in valset.images:
        image_name = image_path.replace('\\', '/').split('/')[-1]
        image_names.append(image_name)

    # Save segmentation results
    result_dir = os.path.join('demo', model_name.lower())
    groundtruth_dir = os.path.join('demo', 'groundtruth')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    for idx, (images, targets) in enumerate(tqdm.tqdm(valloader, desc='Demo')):
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)

        # Load input image
        batch_size = images.shape[0]
        images = [TF.resize(torchvision.io.read_image(image_path).unsqueeze(0), [512, 512], antialias=True)
                  for image_path in valset.images[batch_size * idx:batch_size * (idx + 1)]]
        images = torch.cat(images, dim=0).to(device)
        targets = datasets.utils.draw_segmentation_masks(images, targets, valset.colors)
        outputs = datasets.utils.draw_segmentation_masks(images, outputs, valset.colors)

        # process per 1 batch
        for i in range(targets.shape[0]):
            torchvision.io.write_jpeg(targets[i].cpu(),
                                      os.path.join(groundtruth_dir, image_names[batch_size * idx + i]),
                                      quality=100)
            torchvision.io.write_jpeg(outputs[i].cpu(),
                                      os.path.join(result_dir, image_names[batch_size * idx + i]),
                                      quality=100)
