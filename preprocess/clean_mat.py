from typing import Optional
from scipy.io import loadmat, savemat
import os
import numpy as np
import tifffile

from preprocess.detect_bad_bands import bad_band_mask


FILE_PATH = "../data"


def clean_mat(input_dir: str, output_dir: str):
    """
    A function that clean the mat files from the `input_dir`,
    and save the clean versions in the `output_dir`
    Currently, it does
    - Change the main key to be 'input'
    - Add a key called 'gt' meant to act as a ground_truth
    !!! Ensure gt is not changed when adding noise
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        print(f"Processing {fname}")
        img_path = os.path.join(input_dir, fname)
        img = None

        if fname.endswith(".mat"):
            img = process_mat(img_path)
        elif fname.endswith(".npy"):
            img = process_npy(img_path)
        elif fname.endswith(".tif"):
            img = process_tif(img_path)
        else:
            continue

        if img is None:
            print(f"[!] Warning: No valid array found in {fname}")
            continue
        img = change_shape(img)  # --> (H, W, C)
        print(f"Transposed shape (H, W, C): {img.shape}")

        bad, metrics = bad_band_mask(img)
        img = img[:, :, ~bad]
        print(f"Bad bands: {np.where(bad)[0].tolist()}")
        print(f"Band removed shape (H, W, C): {img.shape}")

        output_fname = os.path.splitext(fname)[0] + ".mat"

        savemat(
            os.path.join(output_dir, output_fname),
            {"input": img, "gt": img},
        )
        print(f"[✓] Cleaned and saved: {fname}")
        print("-----------------")


def process_mat(img_path: str):
    """
    Loads a .mat file, extracts the first valid 3D NumPy array, and transposes it
    from (H, W, C) to (C, H, W) if applicable.

    Parameters:
        img_path (str): Path to the .mat file.

    Returns:
        np.ndarray or None: The transposed array, or None if not found.
    """
    data = loadmat(img_path)

    img = None
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            img = data.get(key)

            # if img is not None and img.ndim == 3:
            #     img = img.transpose(2, 0, 1)
            return img
    print(f"[❌] Error: No valid key found in {img_path}")


def process_npy(img_path: str):
    """
    Loads a .npy file and transposes the array from (H, W, C) to (C, H, W) if 3D.

    Parameters:
        img_path (str): Path to the .npy file.

    Returns:
        np.ndarray: The loaded and possibly transposed array.
    """
    img = np.load(img_path)
    if img.ndim == 3:
        # img = img.transpose(2, 0, 1)
        return img
    print(f"[❌] Error: Array shape is wrong in {img_path}")


def process_tif(img_path: str) -> Optional[np.ndarray]:
    """
    Loads a .tif HSI image and ensures the output shape is (C, H, W).

    Parameters:
        img_path (str): Path to the .tif file.

    Returns:
        np.ndarray: The loaded and shape-corrected HSI array.
    """
    img = tifffile.imread(img_path)
    if img.ndim == 3:
        # (C, H, W) -> (H, W, C)
        img = img.transpose(1, 2, 0)
        return img
    else:
        print(f"[❌] Error: Unexpected array shape {img.shape} in {img_path}")


def change_shape(img: np.ndarray) -> np.ndarray:
    assert img.ndim == 3, "The image must have 3 dimensions"
    print(f"Shape of the image: {img.shape}")
    print("""
        [1]: (H, W, C)  
        [2]: (C, H, W)  
        [3]: (H, C, W)  
        [4]: (W, H, C)  
        [5]: (C, W, H)  
        [6]: (W, C, H)  
    """)
    transpose_choice = input("What is the shape of the image [1-6 or <enter> to skip]: ")
    if transpose_choice != "":
        transpose_choice = int(transpose_choice)
        match transpose_choice:
            case 1:
                print("No need to transpose")
            case 2:
                img = img.transpose(1, 2, 0)
            case 3:
                img = img.transpose(0, 2, 1)
            case 4:
                img = img.transpose(1, 0, 2)
            case 5:
                img = img.transpose(2, 1, 0)
            case 6:
                img = img.transpose(1, 2, 0)
    return img


if __name__ == "__main__":
    print(f"Cleaning the files from {FILE_PATH}")
    clean_mat(f"{FILE_PATH}/raw", f"{FILE_PATH}/clean")
    print("-------------------------------------")
