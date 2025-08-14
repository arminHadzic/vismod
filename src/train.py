import logging
import argparse

from lightning.pytorch.cli import LightningCLI

from utils import setup_logging
from model import ModClassifier
from data import ImageDataModule


def main():
  # parse only logging flags, pass the rest through to LightningCLI
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--log_level", default="INFO")
  parser.add_argument("--log_file", default=None)
  known, remaining = parser.parse_known_args()
  setup_logging(log_level=known.log_level, log_file=known.log_file)
  LightningCLI(
      model_class=ModClassifier,
      datamodule_class=ImageDataModule,
      save_config_callback=None,
      subclass_mode_data=True,
      args=remaining)


if __name__ == "__main__":
  main()
