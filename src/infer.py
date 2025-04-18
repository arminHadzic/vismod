import os
import sys
import numpy as np
import pandas as pd
import imageio.v3 as iio
from pathlib import Path

import torch
from torchvision import transforms
from lightning.pytorch import LightningModule

from model import ModClassifier


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
      transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                           std=[0.229, 0.224, 0.225]),
  ])
  return transform(img).unsqueeze(0)


def setup_model(checkpoint_path: Path = None, device: str = "cpu", verbose: bool = False) -> LightningModule:
  if checkpoint_path:
    if verbose:
      print(f"[•] Loading checkpoint: {checkpoint_path}")
    model = ModClassifier.load_from_checkpoint(checkpoint_path)
  else:
    if verbose:
      print("[•] No checkpoint provided. Using base pretrained ConvNeXtV2 model.")
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
  label = "Not" if prob > threshold else "Safe"
  return {
      "filename": str(image_path),
      "probability": prob,
      "label": label,
  }


def run_parallel_inference(image_paths, checkpoint_path, threshold=0.5, num_workers=4):
  from functools import partial
  from concurrent.futures import ProcessPoolExecutor

  fn = partial(predict_single,
               checkpoint_path=checkpoint_path,
               threshold=threshold)
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
  print(f"[✓] Wrote {len(df)} rows to {output_path}")


def main(checkpoint_path: str,
         input_path: str,
         output_dir: str = "inference_results",
         threshold: float = 0.5,
         num_workers: int = 4,
         batch_size: int = 1000):
  input_path = Path(input_path)
  checkpoint_path = Path(checkpoint_path)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  processed_files = get_already_processed_filenames(output_dir)
  print(f"[•] Found {len(processed_files)} previously processed files")

  if input_path.is_file():
    if str(input_path) in processed_files:
      print(f"[✓] Skipping already processed: {input_path.name}")
      return
    new_result = predict_single(input_path, checkpoint_path, threshold)
    write_partition(pd.DataFrame([new_result]), output_dir, 0)
    return

  if not input_path.is_dir():
    print(f"[!] Invalid input path: {input_path}")
    return

  image_files = (
    Path(entry.path) for entry in os.scandir(input_path)
    if entry.is_file() and Path(entry.name).suffix.lower() in {".jpg", ".jpeg", ".png"}
  )
  to_process = [f for f in image_files if str(f) not in processed_files]

  if not to_process:
    print("[✓] All images already processed.")
    return

  print(f"[•] Processing {len(to_process)} images...")

  for i in range(0, len(to_process), batch_size):
    batch = to_process[i:i + batch_size]
    results = run_parallel_inference(batch, checkpoint_path, threshold, num_workers)
    df = pd.DataFrame(results)
    write_partition(df, output_dir, i // batch_size)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Batch inference with ModClassifier")
  parser.add_argument("--checkpoint", 
                      type=str, 
                      default=None, 
                      help="Path to model .ckpt file (optional)")
  parser.add_argument("--input",
                      type=str,
                      required=True,
                      help="Path to image or directory")
  parser.add_argument("--output",
                      type=str,
                      default="inference_results",
                      help="Output directory (Parquet files)")
  parser.add_argument("--threshold",
                      type=float,
                      default=0.5,
                      help="Classification threshold")
  parser.add_argument("--num_workers",
                      type=int,
                      default=4,
                      help="Number of parallel workers")
  parser.add_argument("--batch_size",
                      type=int,
                      default=1000,
                      help="Number of images per output file")

  args = parser.parse_args()
  main(args.checkpoint, args.input, args.output, args.threshold, args.num_workers,
       args.batch_size)
