import torch
import torch.nn as nn 
from torch import Tensor

class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AvgPool2d(2),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(256, out_channels, 3, 1, 1)
        )

        self.activation = nn.Sigmoid()
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        input = torch.cat((x, y), dim=1)
        x = self.discriminator(input)
        x = self.activation(x)

        return x

