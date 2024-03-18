import numpy as np

from src.data.components.gdbd_dataset import GDBDataset
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Any

class GDBDataModule():
    def __init__(
        self,
        data_root: Path,
        transform: Any,
        batch_size: int = 16,
    ) -> None:
        self.data_root = data_root
        self.transform = transform
        self.batch_size = batch_size

    def setup(self) -> None:
        self.train_dataset = GDBDataset(self.data_root, transfrom=self.transform, split="train")
        self.val_dataset = GDBDataset(self.data_root, transfrom=self.transform, split="val")

    def train_dataloader(self) -> None:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> None:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)