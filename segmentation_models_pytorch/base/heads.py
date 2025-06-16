import torch.nn as nn
from .modules import Activation


# class SegmentationHead(nn.Sequential):
#     def __init__(
#         self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
#     ):
#         conv2d = nn.Conv2d(
#             in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
#         )
#         upsampling = (
#             nn.UpsamplingBilinear2d(scale_factor=upsampling)
#             if upsampling > 1
#             else nn.Identity()
#         )
#         activation = Activation(activation)
#         super().__init__(conv2d, upsampling, activation)

class SegmentationHead(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        super(SegmentationHead, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        self.activation = Activation(activation)

    def forward(self, x):
        # SDAA UpsamplingBilinear2d prec bug
        device = x.device
        return self.activation(self.upsampling(self.conv2d(x).cpu()).to(device))


class ClassificationHead(nn.Sequential):
    def __init__(
        self, in_channels, classes, pooling="avg", dropout=0.2, activation=None
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
