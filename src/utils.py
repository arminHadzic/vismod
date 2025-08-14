import logging
from sys import stdout
from pathlib import Path
from typing import Optional

from torchvision import transforms


def get_image_transform(image_size: int = 224):
  return transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],  # ImageNet
          std=[0.229, 0.224, 0.225]),
  ])


def setup_logging(
    log_file: Optional[str] = None,
    log_level: Optional[str] = "INFO",
    force: bool = True,
) -> None:
  """
  Configure root logging.
  -log_file=None  -> only stdout
  -log_file="path"-> stdout + file
  -log_level: DEBUG|INFO|WARNING|ERROR|CRITICAL
  -force=True reconfigures if logging was already set up
  """
  handlers = [logging.StreamHandler(stdout)]

  if log_file != None:
    log_file_path = Path(log_file)
    # create parent dir if specified (e.g., /app/logs/eval.log)
    if log_file_path.parent.as_posix() not in ("", "."):
      log_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
      handlers.append(logging.FileHandler(log_file_path))
    except FileNotFoundError:
      # fallback to stdout-only if path is invalid
      pass

  logging.basicConfig(
      level=getattr(logging, log_level.upper(), logging.INFO),
      format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
      handlers=handlers,
      force=force,  # ensure reconfig works even if something logged earlier
  )

  # Quiet noisy libs
  logging.getLogger("urllib3").setLevel(logging.WARNING)
  logging.getLogger("PIL").setLevel(logging.WARNING)
