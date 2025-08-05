from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

from model import HSDTLightning


def cli_main():
    cli = LightningCLI(HSDTLightning, BoringDataModule)


if __name__ == "__main__":
    cli_main()
