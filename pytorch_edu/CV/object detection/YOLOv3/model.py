import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, is_bn=True, **kwarg):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not is_bn, **kwarg)
        self.leaky = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn = is_bn

    def forward(self, x):
        if self.use_bn:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)
        

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()

        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_repeats = num_repeats

        for repeats in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    conv_block(channels, channels // 2, kernel_size=1),
                    conv_block(channels // 2, channels, kernel_size=3, padding=1)
                )
            ]
        
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                return x + layer(x)
            else:
                return layer(x)
            

class ScalePrediction(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            conv_block(channels, channels * 2, kernel_size=3, padding=1),
            conv_block(channels * 2, (num_classes + 5) * 3, is_bn=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            )


class YoloV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connection = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                out = layer(x)
                outputs += [out]
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connection += [x]
            
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connection[-1]], dim=1)
                route_connection.pop()

        return outputs
            

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers += [
                    conv_block(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0
                    )
                ]
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers += [
                    ResidualBlock(in_channels, num_repeats=num_repeats)
                ]
            
            elif isinstance(module, str):
                if module == 'S':
                    layers += [
                        ResidualBlock(in_channels, num_repeats=1),
                        conv_block(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == 'U':
                    layers += [nn.Upsample(scale_factor=2)]
                    in_channels = in_channels * 3
        
        return layers
    
# model = YoloV3(num_classes=20).to('cuda')
# x = torch.randn(2, 3, 416, 416).to('cuda')
# print(model(x)[1].shape)
# summary(model, (3, 416, 416))
# print(model)