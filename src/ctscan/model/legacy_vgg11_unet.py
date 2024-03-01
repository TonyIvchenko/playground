from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class Block(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, batch_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm = bool(batch_norm)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        return F.relu(x, inplace=True)


class LegacyVGG11UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        batch_norm: bool = True,
        upscale_mode: str = "bilinear",
        encoder_weights: torchvision.models.VGG11_Weights | None = None,
    ):
        super().__init__()
        self.upscale_mode = str(upscale_mode)
        self.init_conv = nn.Conv2d(in_channels, 3, 1)

        encoder = torchvision.models.vgg11(weights=encoder_weights).features
        self.conv1 = encoder[0]
        self.conv2 = encoder[3]
        self.conv3 = encoder[6]
        self.conv3s = encoder[8]
        self.conv4 = encoder[11]
        self.conv4s = encoder[13]
        self.conv5 = encoder[16]
        self.conv5s = encoder[18]

        self.center = Block(512, 512, 256, batch_norm=batch_norm)
        self.dec5 = Block(512 + 256, 512, 256, batch_norm=batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm=batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm=batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm=batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm=batch_norm)
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    @staticmethod
    def down(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=2)

    def up(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            x,
            size=size,
            mode=self.upscale_mode,
            align_corners=False if self.upscale_mode != "nearest" else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_conv = F.relu(self.init_conv(x), inplace=True)

        enc1 = F.relu(self.conv1(init_conv), inplace=True)
        enc2 = F.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = F.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = F.relu(self.conv3s(enc3), inplace=True)
        enc4 = F.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = F.relu(self.conv4s(enc4), inplace=True)
        enc5 = F.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = F.relu(self.conv5s(enc5), inplace=True)

        center = self.center(self.down(enc5))
        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], dim=1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], dim=1))
        return self.out(dec1)
