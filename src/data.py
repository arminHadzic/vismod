from pathlib import Path

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class ImageDataModule(pl.LightningDataModule):

  def __init__(
      self,
      data_dir: str = "data",
      image_size: int = 224,
      batch_size: int = 32,
      num_workers: int = 4,
  ):
    super().__init__()
    self.data_dir = Path(data_dir)
    self.image_size = image_size
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.save_hyperparameters(ignore=["data_dir"])  # Optional: avoids Path serialization issues

    self.transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]),
    ])

  def setup(self, stage=None):
    train_path = self.data_dir / "train"
    val_path = self.data_dir / "val"
    test_path = self.data_dir / "test"

    if not train_path.exists():
      raise FileNotFoundError(f"[✘] Training folder not found: {train_path}")

    print(f"[✓] Using training data from: {train_path}")

    self.train_dataset = datasets.ImageFolder(train_path, transform=self.transform)

    if val_path.exists():
      self.val_dataset = datasets.ImageFolder(val_path, transform=self.transform)
    elif test_path.exists():
      self.val_dataset = datasets.ImageFolder(test_path, transform=self.transform)
    else:
      print("[!] Validation folder not found — using training data as val split")
      val_size = int(0.2 * len(self.train_dataset))
      train_size = len(self.train_dataset) - val_size
      self.train_dataset, self.val_dataset = torch.utils.data.random_split(
          self.train_dataset, [train_size, val_size])

    if test_path.exists():
      self.test_dataset = datasets.ImageFolder(test_path, transform=self.transform)
    else:
      print("[!] Test folder not found — skipping test set")
      self.test_dataset = None

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True,
        num_workers=self.hparams.num_workers)

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.hparams.batch_size,
        shuffle=False,
        num_workers=self.hparams.num_workers)

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.hparams.batch_size,
        shuffle=False,
        num_workers=self.hparams.num_workers)
