import sys

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner
from matplotlib.figure import Figure

from model import HSDTLightning
from data_module import HSIDataModule


class HSDTLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--run_batch_size_finder", type=bool, default=False, help="Whether to run the batch size finder")
        parser.add_argument(
            "--batch_size_finder_mode", type=str, default="power", help="Mode for batch size finder (power|binsearch)"
        )
        parser.add_argument("--run_lr_finder", type=bool, default=False, help="Whether to run learning rate finder")
        parser.add_argument("--show_lr_plot", type=bool, default=True, help="Whether to plot learning rate finder")

    def before_fit(self):
        tuner = Tuner(self.trainer)

        # ----------------------------------
        # Batch Size Finder
        # ----------------------------------
        # CLI params
        #   run_batch_size_finder (bool): Determines if batch_size finder is ran. Default is True.
        #   batch_size_finder_mode (str): "power" or "binsearch". Determines the mode of batch_size finder
        # ----------------------------------
        if self.config.fit.run_batch_size_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping batch finder due to fast_dev_run")
            else:
                mode = self.config.fit.batch_size_finder_mode
                print(f"\n📦 Running batch size finder (mode: {mode})...")

                new_batch_size = tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode=mode)

                print(f"✅ Suggested batch size: {new_batch_size}")

                if new_batch_size is None:
                    print("⚠️ Could not find optimal batch size")
            exit(0)

        # ----------------------------------
        # Finding Optimal Learning Rate
        # ----------------------------------
        # CLI params
        #   run_lr_finder (bool): Determines if LR finder is ran. Default is True.
        #   show_lr_plot (bool): Determines if LR finder plot is show. Default is False.
        # ----------------------------------
        if self.config.fit.run_lr_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping LR finder due to fast_dev_run")
            else:
                lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

                if lr_finder is not None:
                    if self.config.fit.show_lr_plot:
                        fig = lr_finder.plot(suggest=True)
                        if isinstance(fig, Figure):
                            fig.savefig("logs/lr_finder_plot.png")

                    suggested_lr = lr_finder.suggestion()
                    print(f"\n🔎 Suggested Learning Rate: {suggested_lr:.2e}")
                else:
                    print("⚠️ Could not find optimal learning rate")
            exit(0)


def cli_main():
    argv = sys.argv[1:]
    if argv and argv[0] in {"fit", "validate", "test", "predict", "tune"}:
        command, remainder = argv[0], argv[1:]
        default_args = [command, "--config", "config/base.yaml"]
        args = default_args + remainder
    else:
        args = [
            "--config",
            "config/base.yaml",
            "--config",
            "config/models/tdsat.yaml",
            *argv,
        ]

    HSDTLightningCLI(
        model_class=HSDTLightning,
        datamodule_class=HSIDataModule,
        args=args,
    )


if __name__ == "__main__":
    cli_main()
