# ⚡ HSDT-Lightning
PyTorch Lightning ⚡ implementation of the [HSDT](https://arxiv.org/pdf/2303.09040) model for removing dark noise from hyperspectral images (HSI) — developed for the FINCH satellite by the University of Toronto Aerospace Team.

## 🚀 Features
- ⚡ Built with **PyTorch Lightning** for clean, scalable research
- 🔁 **Multi-GPU training** via Distributed Data Parallel (DDP)
- ⚙️ Fully configurable with **YAML-based CLI interface**
- 📄 **Well-documented**, modular, and strongly-typed codebase

Note that my companion notes can be found in [notion](https://utat-ss.notion.site/HSDT-Denoiser-aa306c141f8c4bbd8100d43efe740df1)

## 📦 Installing Dependencies
Create a virtual environment, and run
```bash
pip install -r requirements.txt
```

## 🗂️ Data Preparation
1. Create a `data` folder.
2. Create `raw` and `test` inside the `data` folder.
3. Download your hyperspectral images to train on into the `raw` folder.
4. Download your hyperspectral images to train on into the `test` folder.
5. (Optional to 6) Run `python -m preprocess.main`. 
6. (Optional to 5) Ensure data: preprocess_data is set to `true` inside the yaml config file

## ⚙️ Running the code
### 🔧 Training
```bash
python main.py fit --config config/train.yaml
```

### ✅ Validation
```bash
python main.py validate --config config/train.yaml
```

### 🧪 Testing
```bash
python main.py test --config config/train.yaml
```

### 🔮 Predict
```bash
python main.py predict --config config/train.yaml
```

### 🆘 For help text
```bash
python main.py --help
```
Note that all the individual commands also have `--help`

## 🛠️ Generating new config file
If you change model or data module parameters, regenerate the config file:

```bash
python main.py fit --print_config > configs/default.yaml
```

Use an existing config (e.g. config/train.yaml) as a template to fill in values.

## 🧾 Project Structure
Overview of the project structure:

```

├── main.py               # Entry point using LightningCLI
├── model.py              # LightningModule for HSDT model
├── data_module.py        # LightningDataModule with transforms
├── dataset.py            # HSI dataset & patching logic
│
├── config/               # YAML configuration files
├── hsdt/                 # HSDT architecture (`hsdt/arch.py`)
├── metrics/              # Metrics: SSIM (`ssim.py`), PSNR (`psnr.py`)
├── preprocess/           # Preprocessing scripts (`main.py` is entry point)
│
├── data/                 # Input images for training/testing
│   ├── raw/              # Raw training data
│   └── test/             # Testing data
│
├── logs/                 # Lightning logs
│
└── checkpoints/
    ├── best/             # Best-performing checkpoints (highest PSNR/SSIM)
    └── interval/         # Checkpoints saved every 5 epochs

```

## 🧰 Technologies Used
- [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) - Training Loop Abstraction
- [Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) - Deep learning framework
- [Scikit Image](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html) - Image quality metrics (SSIM & PSNR)
- [Scipy](https://scipy.org/) - For loading/saving .mat files

For complete list, consult the `requirements.txt`
