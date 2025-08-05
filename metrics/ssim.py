import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_batch_mssim(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the average MSSIM (Mean Structural Similarity Index) over a batch of images.

    This function assumes predictions and targets are PyTorch tensors with shape (B, C, H, W),
    where B is the batch size, C is the number of channels (e.g., spectral bands), and H and W are spatial dimensions.
    It converts each image pair to NumPy and computes MSSIM per sample.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted images, shape (B, C, H, W), values should be float tensors (not integers).

    targets : torch.Tensor
        Ground truth images, same shape as `predictions`.

    Returns
    -------
    float
        The average MSSIM score across the batch.

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
    mssim_scores = []

    for i in range(batch_size):
        pred_img = predictions[i].detach().cpu().permute(1, 2, 0).numpy()  # CHW → HWC
        target_img = targets[i].detach().cpu().permute(1, 2, 0).numpy()

        score = MSSIM(pred_img, target_img)
        mssim_scores.append(score)

    avg_mssim = sum(mssim_scores) / batch_size
    return avg_mssim


def MSSIM(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Mean Structural Similarity Index (MSSIM) between two hyperspectral images.

    This function supports both grayscale and multi-channel (e.g., RGB or hyperspectral) images.
    Images are automatically normalized to the range [0, 1] before computing SSIM.

    Parameters:
    ----------
    img1 : np.ndarray
        First image, with shape (H, W) for grayscale or (H, W, C) for multi-channel/hyperspectral.

    img2 : np.ndarray
        Second image, same shape as `img1`.

    Returns:
    -------
    float
        The average SSIM score across all channels. If input is grayscale, this is the standard SSIM.

    Raises:
    ------
    AssertionError:
        If the shapes of the two input images do not match.

    Notes:
    ------
    - Hyperspectral images typically have many channels, and MSSIM provides a single scalar score
      by averaging SSIM over all spectral bands.
    - Internally uses `skimage.metrics.structural_similarity` for each channel.
    """
    assert img1.shape == img2.shape, "Input images must have the same shape"

    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    if img1.ndim == 2 or (img1.ndim == 3 and img1.shape[2] == 1):
        # Single channel
        ssim_score = ssim(img1.squeeze(), img2.squeeze(), data_range=1.0)
        if isinstance(ssim_score, tuple):
            # Some versions of skimage return (score, ssim_map)
            ssim_score = ssim_score[0]
        return float(ssim_score)

    # Multi-channel (e.g., hyperspectral cube)
    ch = img1.shape[2]
    ssim_total = 0.0
    for i in range(ch):
        ssim_score = ssim(img1[:, :, i], img2[:, :, i], data_range=1.0)
        if isinstance(ssim_score, tuple):
            # Some versions of skimage return (score, ssim_map)
            ssim_score = ssim_score[0]
        ssim_total += float(ssim_score)
    return ssim_total / ch


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].

    Parameters:
    ----------
    img : np.ndarray
        The input image of any shape. Assumes numerical dtype.

    Returns:
    -------
    np.ndarray
        Normalized image with the same shape as input, and values in [0, 1].
    """
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)
