import torch
import torch.nn as nn 
from torch import Tensor

from src.models.components.layers.double_convs import DoubleConvs

import logging as log

logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
console = log.StreamHandler()
console_formater = log.Formatter("[ %(levelname)s ] %(message)s")
console.setFormatter(console_formater)
logger.addHandler(console)

class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features_channels: list[int]
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down part of U-Net
        for channel in features_channels:
            self.downs.append(
                DoubleConvs(in_channels, channel)
            )
            in_channels = channel

        # Up part of U-Net
        for channel in features_channels[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(2*channel, channel, 2, 2)
            )
            self.ups.append(
                DoubleConvs(2*channel, channel)
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConvs(features_channels[-1], 2*features_channels[-1])

        self.conv = nn.Sequential(
            nn.Conv2d(features_channels[0], 64, 3, 1, 1, bias=True),
            nn.SiLU()
        )

        self.final_conv = nn.Conv2d(64, out_channels, 1)

        self.activation = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_tensor = skip_connections[idx//2]
            logger.debug("x`s shape = {}, skep_connection`s shape = {}".format(x.shape, skip_tensor.shape))
            concat_skip = torch.cat((skip_tensor, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        x = self.conv(x)
        x = self.final_conv(x)
        x = self.activation(x)

        return x