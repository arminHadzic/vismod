import shutil
import random
import tarfile
import argparse
from pathlib import Path
from urllib.request import urlretrieve

lbl_dict = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute')

SAFE_WNIDS = {
    "n01440764", "n02102040", "n02979186", "n03028079", "n03394916", "n03445777", "n03888257"
}
NOT_WNIDS = {"n03000684"}

MAX_TOTAL_IMAGES = 1000
TRAIN_RATIO = 0.8
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def download_and_extract_imagenette(target_dir: Path):
  archive = target_dir / "imagenette2-320.tgz"
  extract_dir = target_dir / "imagenette2-320"

  if extract_dir.exists():
    print(f"[✓] Source imagery already exists at {extract_dir}")
    return extract_dir / "train"

  target_dir.mkdir(parents=True, exist_ok=True)

  print(f"Downloading Imagenette to {archive}...")
  urlretrieve(IMAGENETTE_URL, archive)

  print(f"Extracting to {extract_dir}...")
  with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(target_dir)

  return extract_dir / "train"


def relabel_and_sample(src_root: Path):
  print("Relabeling and sampling...")

  all_samples = []

  for wnid_dir in src_root.iterdir():
    if not wnid_dir.is_dir():
      continue

    label = ("safe"
             if wnid_dir.name in SAFE_WNIDS else "not" if wnid_dir.name in NOT_WNIDS else None)
    if label is None:
      continue

    for img_path in wnid_dir.glob("*.JPEG"):
      all_samples.append((img_path, label))

  print(f"Found {len(all_samples)} candidate images...")
  random.seed(42)
  random.shuffle(all_samples)
  all_samples = all_samples[:MAX_TOTAL_IMAGES]

  split_idx = int(len(all_samples) * TRAIN_RATIO)
  train_samples = all_samples[:split_idx]
  test_samples = all_samples[split_idx:]

  return {"train": train_samples, "test": test_samples}


def copy_samples(samples, dest_root: Path):
  for split, entries in samples.items():
    for img_path, label in entries:
      out_dir = dest_root / split / label
      out_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy(img_path, out_dir / img_path.name)


def main():
  parser = argparse.ArgumentParser(description="Prepare a binary proxy dataset using Imagenette.")
  parser.add_argument(
      "--src",
      type=str,
      default="downloads",
      help="Path to source Imagenette root directory (will skip download if it exists)")
  parser.add_argument(
      "--out", type=str, default="data", help="Path to write the output dataset (default: data/)")

  args = parser.parse_args()
  src_path = Path(args.src)
  out_path = Path(args.out)

  imagenette_train_dir = download_and_extract_imagenette(src_path)
  samples = relabel_and_sample(imagenette_train_dir)
  copy_samples(samples, out_path)

  print(f"Dataset ready at: {out_path}/train/, {out_path}/test/")


if __name__ == "__main__":
  main()
