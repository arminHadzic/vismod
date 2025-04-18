from lightning.pytorch.cli import LightningCLI

from model import ModClassifier
from data import ImageDataModule


def main():
  LightningCLI(
      model_class=ModClassifier,
      datamodule_class=ImageDataModule,
      save_config_callback=None,
      subclass_mode_data=True)


if __name__ == "__main__":
  main()
