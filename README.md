# HSDT-Lightning
Implementing the [HSDT](https://github.com/Zeqiang-Lai/HSDT) model using pytorch lightning in order to remove dark noise of hyperspectral images for FINCH satellite of UofT Aerospace Team.

- Enable training on multiple-GPU with distributed data parallel (DDP)
- Control training & testing with yaml config files
- Well-Documented & strongly-typed code

Note that my companion notes can be found in [notion](https://utat-ss.notion.site/HSDT-Denoiser-aa306c141f8c4bbd8100d43efe740df1)

## Installing Dependencies
Create a virtual environment, and run
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Create a `data` folder.
2. Create `raw` and `test` inside the `data` folder.
3. Download your hyperspectral images to train on into the `raw` folder.
4. Download your hyperspectral images to train on into the `test` folder.
5. (Optional to 6) Run `python -m preprocess.main`. 
6. (Optional to 5) Ensure data: preprocess_data is set to `true` inside the yaml config file

## Running the code
### Training
```bash
python main.py fit --config config/train.yaml
```

### Validation
```bash
python main.py validate --config config/train.yaml
```

### Testing
```bash
python main.py test --config config/train.yaml
```

### Predict
```bash
python main.py predict --config config/train.yaml
```

### For help text
```bash
python main.py --help
```
Note that all the individual commands also have `--help`

## Generating new config file
- After making changes to the parameters of the model or data module, you need to regenerate the config file
- You are recommended to look at pre-existing config file to set the values in the newly generated config file
```bash
python main.py fit --print_config > configs/default.yaml
```

## File Structure
Brief explanation of all the files & folders in the codebase.

- `main.py` - Contains code for Lightning CLI
- `model.py` - Contains the LightingModule for the hsdt model
- `data_module.py` - Contains LightningDataModule for the HSI dataset, , as well as transformations to apply onto it
- `dataset.py` - Contains the actual HSI dataset, as well as functions for patching
- `config/` - Folder containing all the config files for running the training/testing
- `hsdt/` - Folder containing the actual HSDT architecture from `hsdt/arch.py`
- `metrics/` - Folder containing `ssim.py` and `psnr.py` for metrics
- `preprocess/` - Folder containing the files used for preprocessing the images. Entry file is `preprocess/main.py`
- `data/` - This is where all the images are stored in.

## Technologies Used
- [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) - You are recommended to go read the docs
- Pytorch
- [Scikit Image](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html) - Used for Metrics of SSIM & PSNR
- [Scipy](https://scipy.org/) - To help load & save .mat files
For full list, look at the `requirements.txt`
