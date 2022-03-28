import timm
import torch
import torch.nn as nn
import torchvision

import models


class DeepLabV3(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super(DeepLabV3, self).__init__()

        # Backbone
        if backbone == 'ResNet50':
            backbone = torchvision.models.resnet50(pretrained=True)
            return_nodes = {'layer4.2.relu_2': 'layer4'}
        elif backbone == 'ResNet101':
            backbone = torchvision.models.resnet101(pretrained=True)
            return_nodes = {'layer4.2.relu_2': 'layer4'}
        elif backbone == 'ResNeSt50':
            backbone = timm.create_model('resnest101e', pretrained=True)
            return_nodes = {'layer4.2.act3': 'layer4'}
        else:
            raise NotImplementedError('Wrong backbone.')
        self.backbone = torchvision.models.feature_extraction.create_feature_extractor(backbone, return_nodes)

        # ASPP
        self.head = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.upsample.size = x.size()[-2:]

        x = self.backbone(x)['layer4']
        x = self.head(x)
        x = self.upsample(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3(backbone='ResNeSt50', num_classes=4).to(device)
    models.test.test_model(model, (1, 3, 720, 1280))
