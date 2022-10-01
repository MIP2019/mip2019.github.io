import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GlobalAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttentionBlock, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding='same')

    def forward(self, inputs):
        x = self.adaptive_avg_pool(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        channel_attention = x * inputs
        x = channel_attention.mean(dim=1, keepdim=True)
        x = torch.sigmoid(x)
        spatial_attention = x * channel_attention
        return spatial_attention


class ResUnit(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int = 64,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        out_planes: int,
        num_units: int = 5,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            conv3x3(in_planes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResUnit(in_planes=planes, planes=planes) for _ in range(num_units - 2)]
        )
        self.tail = nn.Sequential(
            conv3x3(planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.head(x)
        out = self.body(out)
        out = self.tail(out)

        return out


class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Backbone, self).__init__()
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            ResBlock(in_channels, 64, 256),
            ResBlock(256, 64, 256),
            # ResBlock(256, 128, 256),
            ResBlock(256, 128, 512),
            # ResBlock(512, 256, 512),
            ResBlock(512, 256, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layer(x)
        return out


class OverlappedLocalAttention(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OverlappedLocalAttention, self).__init__()
        self.attentions = nn.ModuleList([GlobalAttentionBlock(in_channels) for _ in range(4)])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(in_channels * 4, num_classes)

    def forward(self, inputs):
        heights = [round(inputs.size(2) / 3 * i) for i in range(1, 4)]
        widths = [round(inputs.size(3) / 3 * i) for i in range(1, 4)]
        patches = [
            inputs[:, :, :heights[1], :widths[1]],                      # left top
            inputs[:, :, :heights[1], widths[0]:widths[2]],             # right top
            inputs[:, :, heights[0]:heights[2], :widths[1]],            # left bottom
            inputs[:, :, heights[0]:heights[2], widths[0]:widths[2]],   # right bottom
            ]
        attn_outputs = None
        for i, attention in enumerate(self.attentions):
            if i == 0:
                attn_outputs = attention(patches[i])
                attn_outputs = self.avg_pool(attn_outputs)
                attn_outputs = torch.flatten(attn_outputs, 1)
            else:
                attn_output = attention(patches[i])
                attn_output = self.avg_pool(attn_output)
                attn_output = torch.flatten(attn_output, 1)
                attn_outputs = torch.cat([attn_outputs, attn_output], dim=1)
        out = self.linear(attn_outputs)

        return out


class PyramidAttention(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_classes):
        super(PyramidAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 2, 1),
            nn.BatchNorm2d(mid_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.attn1 = GlobalAttentionBlock(in_channels)
        self.attn2 = GlobalAttentionBlock(mid_channels)
        self.attn3 = GlobalAttentionBlock(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear((in_channels + mid_channels + out_channels), num_classes)

    def forward(self, inputs):
        out2 = self.conv1(inputs)
        out3 = self.conv2(out2)

        out1 = self.attn1(inputs)
        out2 = self.attn2(out2)
        out3 = self.attn3(out3)

        out = []
        for idx, item in enumerate([out1, out2, out3]):
            out.append(torch.flatten(self.avg_pool(item), 1))

        out = torch.cat(out, dim=1)
        out = self.linear(out)

        return out


class Classifier(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        overlapped_attention: OverlappedLocalAttention,
        pyramid_attention: PyramidAttention,
        num_classes: int,
        use_overlapped: bool = True,
        use_pyramid: bool = True,
    ):
        super(Classifier, self).__init__()
        self.use_overlapped = use_overlapped
        self.use_pyramid = use_pyramid
        self.backbone = nn.DataParallel(backbone)
        self.overlapped_attention = overlapped_attention if use_overlapped else nn.Identity()
        self.pyramid_attention = pyramid_attention if use_pyramid else nn.Identity()
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.linear = nn.Linear(backbone.out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        out = self.backbone(inputs)

        out1 = self.overlapped_attention(out)
        out2 = self.pyramid_attention(out)

        if self.use_overlapped and self.use_pyramid:
            out = out1 * out2
        elif self.use_overlapped or self.use_pyramid:
            out = out1 if self.use_overlapped else out2
        else:
            out = self.avg_pool(out)
            out = self.linear(out)

        return out

    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.backbone.module.parameters(), "lr": 1.0 * base_lr},
            {"params": self.overlapped_attention.parameters(), "lr": 1.0 * base_lr},
            {"params": self.pyramid_attention.parameters(), "lr": 1.0 * base_lr},
            {"params": self.linear.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
