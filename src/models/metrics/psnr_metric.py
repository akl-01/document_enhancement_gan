from torch import Tensor
import numpy as np


def psnr(img1: Tensor, img2: Tensor, max_value: int = 1.0) -> int:
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""

    mse = np.mean(img1.detach().cpu().numpy().astype(np.float32) - img2.detach().cpu().numpy().astype(np.float32)) ** 2
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))