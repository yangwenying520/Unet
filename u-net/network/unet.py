import torch
import torch.nn as nn
import torch.nn.functional as F


class two_conv3x3(nn.Module):
    """'(conv => BN => ReLU) * 2"""

    def __init__(self, input_channel, out_channel):
        super(two_conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class max_pool(nn.Module):
    def __init__(self, input_channel, out_channnel):
        super(max_pool, self).__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            two_conv3x3(input_channel, out_channnel)
        )

    def forward(self, x):
        x = self.max_pool(x)
        return x


class up_conv(nn.Module):
    def __init__(self, input_channel, out_channel, bilinear=True):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(input_channel,out_channel, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(input_channel, out_channel, kernel_size=2, stride=2)
        self.conv = two_conv3x3(input_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class out_layer(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(out_layer, self).__init__()
        self.out_layer = nn.Conv2d(input_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.out_layer(x)
        return x


class U_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(U_Net, self).__init__()
        self.inc = two_conv3x3(n_channels, 64)
        self.down1 = max_pool(64, 128)
        self.down2 = max_pool(128, 256)
        self.down3 = max_pool(256, 512)
        self.down4 = max_pool(512, 1024)
        self.up1 = up_conv(1024, 512)
        self.up2 = up_conv(512, 256)
        self.up3 = up_conv(256, 128)
        self.up4 = up_conv(128, 64)
        self.out = out_layer(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        # return torch.sigmoid(x)
        return nn.LogSoftmax(dim=1)(x)
        # return nn.Softmax(dim=1)(x)
