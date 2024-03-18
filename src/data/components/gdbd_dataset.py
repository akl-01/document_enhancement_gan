import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision

from typing import Any
from numpy.typing import NDArray
from pathlib import Path

from src.data.utils.patches_utils import get_patches

import logging as log

logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
console = log.StreamHandler()
console_formater = log.Formatter("[ %(levelname)s ] %(message)s")
console.setFormatter(console_formater)
logger.addHandler(console)

class GDBDataset(Dataset):
    def __init__(
        self, 
        data_root: Path,
        csv_filename: str = 'metadata.csv',
        split: str = 'train',
        transfrom: Any = None
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.csv_path = data_root.joinpath(f"{split}_{csv_filename}")
        self.df = pd.read_csv(self.csv_path, index_col=0)
        self.transfrom = transfrom
        self.post_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), 
             torchvision.transforms.Normalize([0.0], [1.0])])

        self.noisy_patches, self.gt_pathces = self.create_patches()

    def __len__(self) -> int:
        return len(self.gt_pathces)
    
    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        gt_image = self.gt_pathces[idx] / 255.0
        noisy_image = self.noisy_patches[idx] / 255.0

        if self.transfrom:
            noisy_image = self.transfrom(noisy_image)
        
        return self.post_transform(noisy_image), self.post_transform(gt_image)

    def create_patches(self) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        noisy_patches = []
        gt_patches = []

        for idx in range(len(self.df)):
            df_el = self.df.iloc[idx]
            noisy_image = np.asarray(Image.open(self.data_root.joinpath(df_el.noisy_img_path)).convert('L'))
            gt_image = np.asarray(Image.open(self.data_root.joinpath(df_el.gt_img_path)).convert('L'))
        
            noisy_patch, gt_patch = get_patches(noisy_image, gt_image, stride=192)
            logger.debug(f"noisy_pathc`s shape: {noisy_patch.shape}, gt_pathc`s shape: {gt_patch.shape}, {idx}")

            noisy_patches.append(noisy_patch)    
            gt_patches.append(gt_patch)

        noisy_patches = np.concatenate(noisy_patches, axis=0)
        gt_patches = np.concatenate(gt_patches, axis=0)

        return noisy_patches, gt_patches