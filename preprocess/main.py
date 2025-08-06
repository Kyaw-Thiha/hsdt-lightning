from typing import List

from preprocess.clean_mat import clean_mat
from preprocess.add_gaussian_noise import add_gaussian_noise
from preprocess.downsample import load_downsample_save

FILE_PATH = "data"

SPATIAL_TARGET = (-1, -1)
SPATIAL_FACTOR = 64
OUT_BANDS = 81
GAUSSIAN_NOISES = [30, 50, 70]


def preprocess(
    file_path=FILE_PATH,
    spatial_target=(-1, -1),
    spatial_factor=SPATIAL_FACTOR,
    out_bands=OUT_BANDS,
    gaussian_noises: List[int] = [],
):
    print(f"Cleaning the files from {file_path}")
    clean_mat(f"{file_path}/raw", f"{file_path}/clean")
    clean_mat(f"{file_path}/test", f"{file_path}/test_clean")
    print("-------------------------------------")

    print(f"Downsampling the files from {file_path}")
    load_downsample_save(
        f"{file_path}/clean",
        f"{file_path}/clean",
        keys=["input", "gt"],
        spatial_factor=spatial_factor,
        target_size=spatial_target,
        out_bands=out_bands,
    )
    load_downsample_save(
        f"{file_path}/test_clean",
        f"{file_path}/test_clean",
        keys=["input", "gt"],
        spatial_factor=spatial_factor,
        target_size=spatial_target,
        out_bands=out_bands,
    )
    print("-------------------------------------")

    for gaussian_noise in gaussian_noises:
        print(f"Generating noisy images of Sigma-{gaussian_noise}")
        add_gaussian_noise(f"{file_path}/clean", f"{file_path}/gaussian_{gaussian_noise}", snr_db=gaussian_noise, clip=(0, 1))
        add_gaussian_noise(
            f"{file_path}/test_clean", f"{file_path}/test_gaussian_{gaussian_noise}", snr_db=gaussian_noise, clip=(0, 1)
        )
        print("-------------------------------------")


if __name__ == "__main__":
    preprocess(file_path=FILE_PATH, spatial_target=SPATIAL_TARGET, out_bands=OUT_BANDS, gaussian_noises=GAUSSIAN_NOISES)
