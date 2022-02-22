import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models


class EdgeModule(nn.Module):
    def __init__(self, in_feature: list[int], mid_feature=256):
        super(EdgeModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feature[0], mid_feature, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_feature)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_feature[1], mid_feature, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_feature)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_feature[2], mid_feature, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_feature)
        )
        self.conv4 = nn.Conv2d(mid_feature, 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(2 * 3, 2, kernel_size=1)
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3):
        self.upsample.size = x1.size()[-2:]

        edge1 = self.conv4(self.conv1(x1))
        edge2 = self.upsample(self.conv4(self.conv2(x2)))
        edge3 = self.upsample(self.conv4(self.conv3(x3)))
        out = self.conv5(torch.cat([edge1, edge2, edge3], dim=1))
        return out


class PPM(nn.Module):
    def __init__(self, features, out_features):
        super(PPM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + 4 * out_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x):
        self.upsample.size = x.size()[-2:]

        branch1 = self.upsample(self.branch1(x))
        branch2 = self.upsample(self.branch2(x))
        branch3 = self.upsample(self.branch3(x))
        branch4 = self.upsample(self.branch4(x))
        x = self.bottleneck(torch.cat([x, branch1, branch2, branch3, branch4], dim=1))
        return x


class Decoder(nn.Module):
    def __init__(self, in_plane1, in_plane2):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane1, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane2, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, xs, xl):
        self.upsample.size = xl.size()[-2:]

        xs = self.upsample(self.conv1(xs))
        xl = self.conv2(xl)
        out = self.conv3(torch.cat([xs, xl], dim=1))
        return out


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        out -= x
        out = self.relu(self.conv2(out))
        return out


class EAGRModule(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        super(EAGRModule, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d((mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, x, edge):
        batch_size = x.size()[0]
        edge = F.interpolate(edge, x.size()[-2:], mode='nearest')
        edge = F.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(batch_size, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(batch_size, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # Project and graph reason
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))

        return out


class EAGRNet(nn.Module):
    def __init__(self, num_classes):
        super(EAGRNet, self).__init__()
        resnet101 = torchvision.models.resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
        return_nodes = {
            'layer1.2.relu_2': 'layer1',
            'layer2.3.relu_2': 'layer2',
            'layer3.22.relu_2': 'layer3',
            'layer4.2.relu_2': 'layer4',
        }
        self.backbone = torchvision.models.feature_extraction.create_feature_extractor(resnet101, return_nodes)

        self.ppm = PPM(2048, 512)
        self.edge_module = EdgeModule([256, 512, 1024])
        self.eagr_module1 = EAGRModule(512, 128, 4)
        self.eagr_module2 = EAGRModule(256, 64, 4)
        self.decoder = Decoder(512, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x):
        self.upsample.size = x.size()[-2:]

        layer1, layer2, layer3, x = self.backbone(x).values()
        x = self.ppm(x)
        edge = self.edge_module(layer1, layer2, layer3)
        x = self.eagr_module1(x, edge.detach())
        layer1 = self.eagr_module2(layer1, edge.detach())
        x = self.decoder(x, layer1)
        x = self.classifier(x)
        x = self.upsample(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EAGRNet(num_classes=19).to(device)
    models.test.test_model(model, (1, 3, 512, 512), '../runs')
