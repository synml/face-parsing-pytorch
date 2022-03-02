import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models


class ConvBnReLU(nn.Sequential):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__(
            nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBnReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        ca = self.gap(x)
        ca = self.conv_atten(ca)
        ca = self.bn_atten(ca)
        ca = self.sigmoid(ca)
        return x * ca


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        return_nodes = {
            'layer2.1.relu_1': 'layer2',
            'layer3.1.relu_1': 'layer3',
            'layer4.1.relu_1': 'layer4',
        }
        self.resnet = torchvision.models.feature_extraction.create_feature_extractor(resnet18, return_nodes)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = ConvBnReLU(512, 128, ks=1, stride=1, padding=0)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBnReLU(128, 128, ks=3, stride=1, padding=1)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.conv_head16 = ConvBnReLU(128, 128, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x).values()
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        avg = self.gap(feat32)
        avg = self.conv_gap(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode='bilinear', align_corners=True)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode='bilinear', align_corners=True)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='bilinear', align_corners=True)
        feat16_up = self.conv_head16(feat16_up)
        return feat8, feat16_up


# This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Sequential):
    def __init__(self):
        super(SpatialPath, self).__init__(
            ConvBnReLU(3, 64, ks=7, stride=2, padding=3),
            ConvBnReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBnReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBnReLU(64, 128, ks=1, stride=1, padding=0),
        )


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.conv_bn_relu = ConvBnReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        out = self.conv_bn_relu(torch.cat([fsp, fcp], dim=1))
        ca = self.gap(out)
        ca = self.conv1(ca)
        ca = self.relu(ca)
        ca = self.conv2(ca)
        ca = self.sigmoid(ca)
        residual = out * ca
        out += residual
        return out


class Classifier(nn.Sequential):
    def __init__(self, in_chan, mid_chan, num_classes):
        super(Classifier, self).__init__(
            ConvBnReLU(in_chan, mid_chan, ks=3, stride=1, padding=1),
            nn.Conv2d(mid_chan, num_classes, kernel_size=1, bias=False),
        )


class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        # here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.classifier = Classifier(256, 256, num_classes)

    def forward(self, x):
        h, w = x.size()[2:]

        feat_res8, feat_cp8 = self.cp(x)  # use res3b1 feature to replace spatial path feature
        x = self.ffm(feat_res8, feat_cp8)
        x = self.classifier(x)
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiSeNet(19).to(device)
    models.test.test_model(model, (1, 3, 512, 512), '../runs')
