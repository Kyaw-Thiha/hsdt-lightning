from scipy.io import loadmat

# File Source
# https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines

DATA_FILEPATH = "../data/raw/"

file_name_to_key = {
    "Indian_pines.mat": "indian_pines",
    "Salinas.mat": "salinas",
    "Pavia.mat": "pavia",
    "PaviaU.mat": "paviaU",
    "Cuprite.mat": "cuprite",
}


def analyze_mat_file(file_path: str, file_name: str):
    print(f"Analyzing the {file_name}")
    data = loadmat(f"{file_path}{file_name}")
    print(f"Keys: {data.keys()}")

    for key in data.keys():
        if not key.startswith("__"):
            cube = data[key]
            print("Shape: ", cube.shape)


if __name__ == "__main__":
    print(f"Analyzing the files from {DATA_FILEPATH}")
    print("*********************************************")
    analyze_mat_file(DATA_FILEPATH, "Indian_pines.mat")
    print("-----------------------------------------------")
    analyze_mat_file(DATA_FILEPATH, "Salinas.mat")
    print("-----------------------------------------------")
    analyze_mat_file(DATA_FILEPATH, "Pavia.mat")
    print("-----------------------------------------------")
    analyze_mat_file(DATA_FILEPATH, "PaviaU.mat")
    print("-----------------------------------------------")
    # analyze_mat_file(DATA_FILEPATH, "Cuprite.mat")
