from lightning.pytorch.cli import LightningCLI

from model import HSDTLightning
from data_module import HSIDataModule


def cli_main():
    cli = LightningCLI(HSDTLightning, HSIDataModule)


if __name__ == "__main__":
    cli_main()
