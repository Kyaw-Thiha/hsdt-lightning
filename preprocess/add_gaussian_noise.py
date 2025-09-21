import os
from typing import Tuple
import numpy as np
from scipy.io import loadmat, savemat

FILE_PATH = "../data"


def add_gaussian_noise(
    input_folder: str, output_folder: str, sigma: float = -1.0, snr_db: int = -1, clip: Tuple[int, int] = (-1, -1)
):
    """
    Add Gaussian noise to all .mat files in a folder.

    Parameters:
        input_folder (str): Path to folder containing clean .mat files.
        output_folder (str): Path to save noisy .mat files.
        sigma (float): Standard deviation of Gaussian noise.
        snr_db (int): Signal to Noise ratio in dB units
        clip (Tuple(int, int): Will clip the noised image to (clip[0], clip[1]).
    """
    assert sigma > 0 or snr_db > 0, f"Either sigma or snr_db must be positive. Received Sigma: {sigma} and SNR: {snr_db}"

    # Since we normalized our image data [0-1], we need to normalize the noise too
    if sigma > 1:
        sigma = sigma / 255

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".mat"):
            continue

        input_path = os.path.join(input_folder, filename)
        data = loadmat(input_path)

        noisy_data = {}
        for key, value in data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and key != "gt":
                ## If sigma is 0, we will use snr_db and estimate sigma value
                local_sigma = sigma
                if sigma < 0:
                    local_sigma = estimate_sigma_from_snr(value, snr_db)
                print(f"Noise Level: {local_sigma}")

                # Normalizing noise strength over each bands
                H, W, C = value.shape
                for band in range(C):
                    img_at_band = value[:, :, band]
                    low = np.percentile(img_at_band, 2.0)
                    high = np.percentile(img_at_band, 98.0)
                    scale = np.maximum(high - low, 1e-12)

                    band_sigma = local_sigma * scale
                    noise = np.random.normal(0, band_sigma, size=(H, W))
                    value[:, :, band] = img_at_band + noise

                clip_lower, clip_upper = clip
                if clip_lower != -1 and clip_upper != -1:
                    noisy_data[key] = np.clip(value, clip_lower, clip_upper)
                else:
                    noisy_data[key] = value
            else:
                # Keep non-array and ground truth data untouched
                noisy_data[key] = value

        output_path = os.path.join(output_folder, filename)
        savemat(output_path, noisy_data)
        print(f"[✓] Noised and saved: {filename}")


def estimate_sigma_from_snr(signal, snr_db: int):
    """
    Estimate noise sigma based on desired SNR in dB.
    """
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power)
    return sigma


if __name__ == "__main__":
    print("Generating noisy images of Sigma-30")
    add_gaussian_noise(f"{FILE_PATH}/clean", f"{FILE_PATH}/noise_gaussian_30", snr_db=30)
    print("-------------------------------------")

    print("Generating noisy images of Sigma-50")
    add_gaussian_noise(f"{FILE_PATH}/clean", f"{FILE_PATH}/noise_gaussian_50", snr_db=50)
    print("-------------------------------------")

    print("Generating noisy images of Sigma-70")
    add_gaussian_noise(f"{FILE_PATH}/clean", f"{FILE_PATH}/noise_guassian_70", snr_db=70)
    print("-------------------------------------")
