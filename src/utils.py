import logging
from sys import stdout
from typing import Optional


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
    handlers.append(logging.FileHandler(log_file))

  logging.basicConfig(
      level=getattr(logging, log_level.upper(), logging.INFO),
      format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
      handlers=handlers,
      force=force,  # ensure reconfig works even if something logged earlier
  )

  # Quiet noisy libs
  logging.getLogger("urllib3").setLevel(logging.WARNING)
  logging.getLogger("PIL").setLevel(logging.WARNING)