import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import models.sfnet_module.module
import models.sfnet_module.resnet_d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()

        self.ppm = models.sfnet_module.module.PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    models.sfnet_module.module.AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(fpn_feature_list[i], output_size, mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


class AlignNetResNet(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D',
                 skip='m1', skip_num=48, fpn_dsn=False):
        super(AlignNetResNet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == trunk == 'resnet-50-deep':
            resnet = models.sfnet_module.resnet_d.resnet50()
        elif trunk == 'resnet-101-deep':
            resnet = models.sfnet_module.resnet_d.resnet101()
        elif trunk == 'resnet-18-deep':
            resnet = models.sfnet_module.resnet_d.resnet18()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        if trunk == 'resnet-18-deep':
            inplane_head = 512
            self.head = UperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                         fpn_inplanes=[64, 128, 256, 512], fpn_dim=128, fpn_dsn=fpn_dsn)
        else:
            inplane_head = 2048
            self.head = UperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                         fpn_dsn=fpn_dsn)

    def forward(self, x):
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        x, _ = self.head([x1, x2, x3, x4])
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x


def DeepR101_SF_deeply(num_classes, criterion=None):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1')


def DeepR50_SF_deeply(num_classes, criterion=None):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')


def DeepR18_SF_deeply(num_classes, criterion=None):
    """
    ResNet-18 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-18-deep', criterion=criterion, variant='D', skip='m1')


def sfnet_impl(backbone: str, num_classes: int):
    if backbone == 'ResNet18':
        model = DeepR18_SF_deeply(num_classes)
    elif backbone == 'ResNet50':
        model = DeepR50_SF_deeply(num_classes)
    elif backbone == 'ResNet101':
        model = DeepR101_SF_deeply(num_classes)
    else:
        raise NotImplementedError('Wrong backbone.')
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = sfnet_impl(backbone='ResNet18', num_classes=4).to(device)
    models.test.test_model(model, (1, 3, 720, 1280))
