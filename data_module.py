import random
from typing import List, Optional
import lightning as L

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from preprocess.main import preprocess
from dataset import HSIDataset


class HSIDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for Hyperspectral Image (HSI) denoising task.

    This module can either load preprocessed data from disk or dynamically generate it
    using the specified preprocessing pipeline.

    Args:
        base_dir (str): Path to the base directory where all HSI folders are stored.
        spatial_factor (int): Desired spatial size for output or patches.
        out_bands (int): Number of spectral bands to keep in the output.
        preprocess_data (bool): Whether to dynamically generate data using `preprocess()` function.
        gaussian_noises (List[int]): List of Gaussian noise levels used (e.g., [30, 50, 70]).
        patch_test (bool): Whether to apply patch extraction to test dataset.
        batch_size (int): Number of samples per batch.
    """

    def __init__(
        self,
        base_dir: str,
        spatial_factor: int = 64,
        out_bands: int = 81,
        preprocess_data: bool = False,
        gaussian_noises: List[int] = [30, 50, 70],
        patch_test: bool = True,
        batch_size: int = 32,
    ):
        super().__init__()
        self.base_dir: str = base_dir
        self.spatial_factor: int = spatial_factor
        self.out_bands: int = out_bands
        self.preprocess_data = preprocess_data
        self.gaussian_noises: List[int] = gaussian_noises
        self.patch_test: bool = patch_test
        self.batch_size: int = batch_size

        # Internal dataset handles
        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        If dynamic generation is enabled, run the preprocessing function once.
        Only runs on rank 0 in distributed setups.
        """
        if self.preprocess_data:
            preprocess(
                file_path=self.base_dir,
                spatial_factor=self.spatial_factor,
                out_bands=self.out_bands,
                gaussian_noises=self.gaussian_noises,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for different stages.

        Args:
            stage (Optional[str]): One of ["fit", "validate", "test", "predict"].
        """
        base_dir = "data"
        patch_size = 64

        if stage == "fit" or stage is None:
            transform = HSITransform(crop_size=64)
            train_dataset = HSIDataset(base_dir, patch_size=patch_size, stride_size=patch_size // 4, transform=transform)
            for gaussian_noise in self.gaussian_noises:
                train_dataset.add_files(f"gaussian_{gaussian_noise}")

            total = len(train_dataset)
            train_size = int(0.9 * total)
            val_size = total - train_size

            self.dataset_train, self.dataset_val = random_split(train_dataset, [train_size, val_size])

        elif stage == "test":
            test_patch_size = None
            if self.patch_test:
                test_patch_size = patch_size
            test_dataset = HSIDataset(base_dir, patch_size=test_patch_size, stride_size=test_patch_size)
            for gaussian_noise in self.gaussian_noises:
                test_dataset.add_files(f"test_gaussian_{gaussian_noise}")

            self.dataset_test = test_dataset

    def train_dataloader(self) -> DataLoader:
        if self.dataset_train is None:
            self.dataset_train = Dataset()
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.dataset_val is None:
            self.dataset_val = Dataset()
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        if self.dataset_test is None:
            self.dataset_test = Dataset()
        return DataLoader(self.dataset_test, batch_size=self.batch_size)


class HSITransform:
    """
    Apply random augmentations to a hyperspectral (C, H, W) tensor:
    - Random horizontal flip
    - Random vertical flip
    - Random 90° rotation
    - Random crop to a fixed size

    Args:
        crop_size (int): Desired crop size (crop_size x crop_size).
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 3, "Expected tensor of shape (C, H, W)"
        C, H, W = tensor.shape
        assert H >= self.crop_size and W >= self.crop_size, "Image too small for cropping."

        # Random horizontal flip
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[2])  # Flip width

        # Random vertical flip
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[1])  # Flip height

        # Random 90° rotation (0, 90, 180, 270 degrees)
        k = random.choice([0, 1, 2, 3])
        tensor = torch.rot90(tensor, k=k, dims=[1, 2])

        # Random crop
        top = random.randint(0, H - self.crop_size)
        left = random.randint(0, W - self.crop_size)
        tensor = tensor[:, top : top + self.crop_size, left : left + self.crop_size]

        return tensor
