import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch


def compute_batch_mpsnr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the average MPSNR (Mean Peak Signal-to-Noise Ratio) over a batch of images.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted images, shape (B, C, H, W), values expected in [0, 1].

    targets : torch.Tensor
        Ground truth images, same shape as `predictions`.

    Returns
    -------
    float
        The average MPSNR score across the batch.

    Raises
    ------
    AssertionError
        If input tensors do not have the same shape or are not 4D tensors.
    """
    assert predictions.shape == targets.shape, (
        "Predictions and targets must have the same shape"
    )
    assert predictions.ndim == 4, "Input tensors must be 4-dimensional (B, C, H, W)"

    batch_size = predictions.size(0)
    mpsnr_scores = []

    for i in range(batch_size):
        pred_img = predictions[i].detach().cpu().permute(1, 2, 0).numpy()
        target_img = targets[i].detach().cpu().permute(1, 2, 0).numpy()

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
    assert img1.shape == img2.shape, "Input images must have the same shape"

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if img1.ndim == 2 or (img1.ndim == 3 and img1.shape[2] == 1):
        return psnr(img1.squeeze(), img2.squeeze(), data_range=1.0)

    ch = img1.shape[2]
    psnr_total = 0.0
    for i in range(ch):
        score = psnr(img1[:, :, i], img2[:, :, i], data_range=1.0)
        psnr_total += score
    return psnr_total / ch
