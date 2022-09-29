from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Callable = None,
        train_dataloader_params: dict = None,
        train_collate_fn: Callable = None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_dataloader_params = train_dataloader_params
        self.train_collate_fn = train_collate_fn

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        # Assign train set for use in dataloader
        if stage in (None, "fit"):
            self.train_set = self.train_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            collate_fn=self.train_collate_fn,
            **self.train_dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
        return None

    def test_dataloader(self) -> DataLoader:
        return None
