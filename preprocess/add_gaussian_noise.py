import os
import numpy as np
from scipy.io import loadmat, savemat

FILE_PATH = "../data"


def add_gaussian_noise(input_folder: str, output_folder: str, snr_db: int):
    """
    Add Gaussian noise to all .mat files in a folder.

    Parameters:
        input_folder (str): Path to folder containing clean .mat files.
        output_folder (str): Path to save noisy .mat files.
        # sigma (float): Standard deviation of Gaussian noise.
        snr_db (int): Signal to Noise ratio in dB units
    """

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
                sigma = estimate_sigma_from_snr(value, snr_db)
                noise = np.random.normal(0, sigma, size=value.shape)
                noisy_data[key] = value + noise
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
    add_gaussian_noise(
        f"{FILE_PATH}/clean", f"{FILE_PATH}/noise_gaussian_30", snr_db=30
    )
    print("-------------------------------------")

    print("Generating noisy images of Sigma-50")
    add_gaussian_noise(
        f"{FILE_PATH}/clean", f"{FILE_PATH}/noise_gaussian_50", snr_db=50
    )
    print("-------------------------------------")

    print("Generating noisy images of Sigma-70")
    add_gaussian_noise(
        f"{FILE_PATH}/clean", f"{FILE_PATH}/noise_guassian_70", snr_db=70
    )
    print("-------------------------------------")
