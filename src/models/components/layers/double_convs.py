import torch.nn as nn 
from torch import Tensor

from typing import Union
from torch.nn.common_types import _size_2_t

class DoubleConvs(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return (self.convs(x) + self.downsample(x)) / 1.4