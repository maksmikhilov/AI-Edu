import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.downs += [DoubleConv(in_channels, channel)]
            in_channels = channel

        for channel in reversed(channels):
            self.ups += [
                nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2),
                DoubleConv(channel * 2, channel)
            ]

        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        print(self.ups)

    def forward(self, x):
        skip_connections = []
        for layer in self.downs:
            x = layer(x)
            skip_connections += [x]
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            print(skip_connection.shape, x.shape)

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


model = UNET()
x = torch.randn(2, 3, 416, 416)
print(model(x).shape)
