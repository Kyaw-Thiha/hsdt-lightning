import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch


def compute_batch_mpsnr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the average MPSNR (Mean Peak Signal-to-Noise Ratio) over a batch of images.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted images, shape (B, C, D, H, W), values expected in [0, 1].

    targets : torch.Tensor
        Ground truth images, same shape as `predictions`.

    Returns
    -------
    float
        The average MPSNR score across the batch.

    Raises
    ------
    AssertionError
        If input tensors do not have the same shape or are not 5D tensors.
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    assert predictions.ndim == 5, f"Expected input tensors to be 5D (B, C, D, H, W), but got {predictions.shape}"

    batch_size = predictions.size(0)
    mpsnr_scores = []

    # Looping over each batch
    # Since we know we only have 1 channel for HSI, we take the first element
    for i in range(batch_size):
        pred_img = predictions[i, 0].detach().cpu().permute(1, 2, 0).numpy()
        target_img = targets[i, 0].detach().cpu().permute(1, 2, 0).numpy()

        score = MPSNR(pred_img, target_img)
        mpsnr_scores.append(score)

    avg_mpsnr = sum(mpsnr_scores) / batch_size
    return avg_mpsnr


def MPSNR(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Mean Peak Signal-to-Noise Ratio (MPSNR) between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image (ground truth), shape (H, W) or (H, W, C), values expected in [0, 1].

    img2 : np.ndarray
        Second image (predicted), same shape as `img1`.

    Returns
    -------
    float
        The mean PSNR across all channels.

    Raises
    ------
    AssertionError
        If input images do not have the same shape.
    """
    PSNR_CAP = 80  # Large finite cap for infinite values
    assert img1.shape == img2.shape, "Input images must have the same shape"

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 2 or (img1.ndim == 3 and img1.shape[2] == 1):
        return psnr(img1.squeeze(), img2.squeeze(), data_range=1.0)

    ch = img1.shape[2]
    psnr_total = 0.0
    for i in range(ch):
        # If two images are equal at the spectral band, return the max PSNR value
        if np.array_equal(img1[:, :, i], img2[:, :, i]):
            score = PSNR_CAP
            psnr_total += score
            continue

        # Calculate the PSNR, and cap the infinite result
        score = psnr(img1[:, :, i], img2[:, :, i], data_range=1.0)
        if not np.isfinite(score):
            score = PSNR_CAP
        psnr_total += score
    return psnr_total / ch
