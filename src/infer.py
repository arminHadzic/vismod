import time
import logging
import numpy as np
import pandas as pd
import imageio.v3 as iio
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import torch
from torchvision import transforms
from lightning.pytorch import LightningModule

from utils import setup_logging
from model import ModClassifier

logger = logging.getLogger(__name__)


def load_image(path, image_size=224):
  img = iio.imread(path)

  # Handle grayscale and RGBA images
  if img.ndim == 2:  # grayscale → [H, W]
    img = np.stack([img] * 3, axis=-1)  # → [H, W, 3]
  elif img.shape[2] == 4:  # RGBA
    img = img[:, :, :3]  # strip alpha

  transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],  # ImageNet
          std=[0.229, 0.224, 0.225]),
  ])
  return transform(img).unsqueeze(0)


def setup_model(checkpoint_path: Path = None, device: str = "cpu") -> LightningModule:
  if checkpoint_path:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = ModClassifier.load_from_checkpoint(checkpoint_path)
  else:
    logger.info("No checkpoint provided. Using base pretrained ConvNeXtV2 model.")
    model = ModClassifier()  # Loads with default model_name from constructor
  model.eval()
  model.freeze()
  return model.to(device)


@torch.no_grad()
def predict_single(image_path: Path, checkpoint_path: Path, threshold: float = 0.5):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = setup_model(checkpoint_path, device)
  image_tensor = load_image(image_path).to(device)
  logits = model(image_tensor).squeeze()
  prob = torch.sigmoid(logits).item()
  pred_class = "Safe" if prob > threshold else "Not"
  return {
      "filename": str(image_path),
      "probability": prob,
      "pred_class": pred_class,
  }


def run_parallel_inference(image_paths, checkpoint_path, threshold=0.5, num_workers=4):
  fn = partial(predict_single, checkpoint_path=checkpoint_path, threshold=threshold)
  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    return list(executor.map(fn, image_paths))


def get_already_processed_filenames(output_dir: Path) -> set:
  if not output_dir.exists():
    return set()

  files = list(output_dir.glob("*.parquet"))
  if not files:
    return set()

  dfs = [pd.read_parquet(f, columns=["filename"]) for f in files]
  combined = pd.concat(dfs, ignore_index=True)
  return set(combined["filename"].tolist())


def write_partition(df: pd.DataFrame, output_dir: Path, part_index: int):
  output_path = output_dir / f"part_{part_index:05d}.parquet"
  df.to_parquet(output_path, index=False)
  logger.info(f"Wrote {len(df)} rows to {output_path}")


def main(checkpoint_path: str,
         input_path: str,
         output_dir: str = "inference_results",
         threshold: float = 0.5,
         num_workers: int = 4,
         samples_per_file: int = 1000,
         log_file: str | None = None,
         log_level: str = "INFO"):

  setup_logging(log_file, log_level)

  input_path = Path(input_path)
  checkpoint_path = Path(checkpoint_path) if checkpoint_path != None else None
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  processed_files = get_already_processed_filenames(output_dir)
  logger.info(f"Found {len(processed_files)} previously processed files")

  if input_path.is_file():
    if str(input_path) in processed_files:
      logger.info(f"Skipping already processed: {input_path.name}")
      return

    new_result = predict_single(input_path, checkpoint_path, threshold)
    write_partition(pd.DataFrame([new_result]), output_dir, 0)
    return

  if not input_path.is_dir():
    logger.error(f"Invalid input path: {input_path}")
    return

  image_files = (
      p for p in Path(input_path).rglob("*")
      if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
  to_process = [p for p in image_files if str(p) not in processed_files]

  if not to_process:
    logger.info("All images already processed.")
    return

  logger.info(f"Processing {len(to_process)} images.")

  for i in range(0, len(to_process), samples_per_file):
    batch = to_process[i:i + samples_per_file]
    t0 = time.time()
    results = run_parallel_inference(batch, checkpoint_path, threshold, num_workers)
    dt = time.time() - t0
    logger.info(f"Batch {i // samples_per_file} processed in {dt:.2f}.")
    df = pd.DataFrame(results)
    write_partition(df, output_dir, i // samples_per_file)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Batch inference with ModClassifier")
  parser.add_argument(
      "--checkpoint", type=str, default=None, help="Path to model .ckpt file (optional)")
  parser.add_argument("--input", type=str, required=True, help="Path to image or directory")
  parser.add_argument(
      "--output", type=str, default="inference_results", help="Output directory (Parquet files)")
  parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
  parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
  parser.add_argument(
      "--samples_per_file", type=int, default=1000, help="Number of images per output file")
  parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
  parser.add_argument("--log_level", default="INFO")

  args = parser.parse_args()
  main(args.checkpoint, args.input, args.output, args.threshold, args.num_workers,
       args.samples_per_file, args.log_file, args.log_level)
