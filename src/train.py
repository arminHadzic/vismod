import logging

from lightning.pytorch.cli import LightningCLI

from utils import setup_logging
from model import ModClassifier
from data import ImageDataModule


def main():
  setup_logging()
  LightningCLI(
      model_class=ModClassifier,
      datamodule_class=ImageDataModule,
      save_config_callback=None,
      subclass_mode_data=True)


if __name__ == "__main__":
  main()
