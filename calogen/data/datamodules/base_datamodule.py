
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    """Base class for Lightning DataModules.

    This class provides a common interface for all DataModules in the project.
    It defines the basic structure and methods that all DataModules should implement.
    """

    def __init__(self, 
        batch_size: int = 32,
        predict_batch_size: int = 1000,
        num_workers: int = 0,
        shuffle: bool = True,
        predict_multiplier: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.predict_batch_size = predict_batch_size
        self.predict_multiplier = predict_multiplier

    def _dl_kwargs(self, shuffle: bool = False, batch_size: int | None = None) -> dict:
        batch_size = batch_size or self.batch_size
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self):
        return DataLoader(self._train_dataset, **self._dl_kwargs(shuffle=self.shuffle))

    def val_dataloader(self):
        return DataLoader(self._val_dataset, **self._dl_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            **self._dl_kwargs(shuffle=False, batch_size=self.predict_batch_size)
        )

    def predict_dataloader(self):

        assert self._test_dataset is not None, (
            "Call setup('predict') before requesting predict_dataloader"
        )
        return [DataLoader(
            self._test_dataset,
            **self._dl_kwargs(shuffle=False, batch_size=self.predict_batch_size)
        ) for _ in range(self.predict_multiplier)]

    def teardown(self, stage=None):
        """Clean up HDF5 handles opened by this DataModule."""
        for attr in ("_train_dataset", "_val_dataset", "_test_dataset"):
            ds = getattr(self, attr, None)
            if ds is not None and hasattr(ds, 'source') and ds.source is not None:
                ds.source.close()