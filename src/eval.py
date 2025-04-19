import argparse
import pandas as pd
from pathlib import Path

import torch
from lightning import seed_everything
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall,
                                         BinaryF1Score, BinaryAUROC)

from model import ModClassifier
from data import ImageDataModule


@torch.no_grad()
def evaluate(checkpoint_path: str, data_dir: str, batch_size: int = 32, num_workers: int = 4):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[•] Loading model from checkpoint: {checkpoint_path}")
  model = ModClassifier.load_from_checkpoint(checkpoint_path)
  model.eval()
  model.to(device)

  print(f"[•] Loading test data from: {data_dir}/test/")
  data_module = ImageDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
  data_module.setup(stage="test")
  test_loader = data_module.test_dataloader()

  acc = BinaryAccuracy().to(device)
  prec = BinaryPrecision().to(device)
  recall = BinaryRecall().to(device)
  f1 = BinaryF1Score().to(device)
  auc = BinaryAUROC().to(device)

  all_probs = []
  all_labels = []

  for batch in test_loader:
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x).squeeze(1)
    probs = torch.sigmoid(logits)

    all_probs.append(probs)
    all_labels.append(y)

  probs = torch.cat(all_probs)
  labels = torch.cat(all_labels)

  acc_val = acc(probs, labels).item()
  prec_val = prec(probs, labels).item()
  recall_val = recall(probs, labels).item()
  f1_val = f1(probs, labels).item()
  auc_val = auc(probs, labels).item()

  print("\n[✓] Evaluation Metrics:")
  print(f"Accuracy:  {acc_val:.4f}")
  print(f"Precision: {prec_val:.4f}")
  print(f"Recall:    {recall_val:.4f}")
  print(f"F1 Score:  {f1_val:.4f}")
  print(f"AUC:       {auc_val:.4f}")

  results_path = Path("eval_results.csv")
  row = {
      "checkpoint": str(checkpoint_path),
      "data_dir": str(data_dir),
      "accuracy": acc_val,
      "precision": prec_val,
      "recall": recall_val,
      "f1_score": f1_val,
      "auc": auc_val,
  }
  df = pd.DataFrame([row])
  if results_path.exists():
    df.to_csv(results_path, mode="a", header=False, index=False)
  else:
    df.to_csv(results_path, index=False)
  print(f"[✓] Logged results to: {results_path}")


def main():
  parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
  parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
  parser.add_argument("--data_dir", required=True, help="Path to dataset root")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--num_workers", type=int, default=4)

  args = parser.parse_args()
  seed_everything(0)
  evaluate(args.checkpoint, args.data_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
  main()
