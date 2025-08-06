import os
from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


class HSIDataset(Dataset):
    """
    Dataset for loading hyperspectral image pairs (noisy and clean) from `.mat` files.

    Each file is expected to have:
    - 'gt': the noisy hyperspectral image
    - 'input': the clean hyperspectral image

    Args:
        base_dir (str): Path to folder containing subfolders with `.mat` files.
        transform (Callable, optional): Transformations to apply to both noisy and clean images.
        patch_size (Optional[int]): If set, will extract random patches of this spatial size.
        stride_size (Optional[int]): Only works if patch_size is set, will extract random patches of this spatial size.
    """

    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        patch_size: Optional[int] = None,
        stride_size: Optional[int] = None,
    ):
        self.base_dir = base_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patches: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add_files(self, folder_name: str):
        data_dir = os.path.join(self.base_dir, folder_name)
        mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
        mat_files.sort()
        files = [os.path.join(data_dir, f) for f in mat_files]

        for mat_file in files:
            mat = loadmat(mat_file)

            noisy = mat["input "]  # Noisy image
            clean = mat["gt"]  # Clean image

            # Ensure (C, H, W) format if needed
            if noisy.ndim == 3:
                # From (H, W, C) to (C, H, W)
                noisy = np.transpose(noisy, (2, 0, 1))
                clean = np.transpose(clean, (2, 0, 1))

                noisy_tensor = torch.tensor(noisy, dtype=torch.float32)
                clean_tensor = torch.tensor(clean, dtype=torch.float32)

                # List of P tensors with shape (C, H, W)
                if self.patch_size is not None and self.stride_size is not None:
                    noisy_tensor = extract_patches(noisy_tensor, self.patch_size, self.stride_size)
                    clean_tensor = extract_patches(clean_tensor, self.patch_size, self.stride_size)
                    noisy_tensor = list(noisy_tensor.unbind(dim=0))
                    clean_tensor = list(clean_tensor.unbind(dim=0))
                else:
                    noisy_tensor = [noisy_tensor]
                    clean_tensor = [clean_tensor]

                # Creating list of tuples of (noisy_tensor, clean_tensor)
                paired_patches = list(zip(noisy_tensor, clean_tensor))
                self.patches.extend(paired_patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        noisy_tensor, clean_tensor = self.patches[idx]

        # Add channel dimension
        noisy_tensor = noisy_tensor.unsqueeze(0)  # (1, C, H, W)
        clean_tensor = clean_tensor.unsqueeze(0)

        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)
            clean_tensor = self.transform(clean_tensor)

        return noisy_tensor, clean_tensor


def extract_patches(tensor: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """
    Extracts non-overlapping patches from an input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W].
        patch_size (int): Size of the square patch.
        stride (int): Stride used for unfolding.

    Returns:
        torch.Tensor: Patches of shape [N, C, patch_size, patch_size], where N is the number of patches.
    """
    C, H, W = tensor.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image too small for patch size {patch_size}: got {H}x{W}")

    patches = tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # [num_patches, C, H, W]
    return patches


def reconstruct_from_patches(
    patches: torch.Tensor, image_shape: Tuple[int, int, int], patch_size: int, stride: int
) -> torch.Tensor:
    """
    Reconstructs an image from patches using averaging in overlapping regions.

    Args:
        patches (torch.Tensor): Tensor of patches [N, C, patch_size, patch_size].
        image_shape (Tuple[int, int, int]): Shape of the original image (C, H, W).
        patch_size (int): Size of the square patch.
        stride (int): Stride used during extraction.

    Returns:
        torch.Tensor: Reconstructed image of shape [C, H, W].
    """
    C, H, W = image_shape
    output = torch.zeros((C, H, W), device=patches.device)
    count = torch.zeros((C, H, W), device=patches.device)

    patch_idx = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            output[:, y : y + patch_size, x : x + patch_size] += patches[patch_idx]
            count[:, y : y + patch_size, x : x + patch_size] += 1
            patch_idx += 1

    return output / count.clamp(min=1)
