import re
import logging
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from lightning import seed_everything
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_precision_recall_curve
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall,
                                         BinaryF1Score, BinaryAUROC, BinaryAveragePrecision)

from utils import setup_logging
from model import ModClassifier
from data import ImageDataModule

logger = logging.getLogger(__name__)

# ---------- Reporting helpers ----------


def save_pr_curve(probs: torch.Tensor, labels: torch.Tensor, out_path: Path, ap_val: float) -> Path:
  """Save a Precision–Recall curve PNG to out_path."""
  pr_prec, pr_rec, _ = binary_precision_recall_curve(probs, labels)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  plt.figure(figsize=(5.5, 4.5))
  plt.plot(pr_rec.numpy(), pr_prec.numpy(), linewidth=2)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(f"Precision–Recall (AP={ap_val:.3f})")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()
  logger.info(f"Saved PR curve: {out_path}")
  return out_path


def render_perf_section(best_thr: float,
                        acc: float,
                        prec: float,
                        rec: float,
                        f1: float,
                        auc: float,
                        ap: float,
                        pr_curve_relpath: str = "assets/pr_curve.png",
                        include_fallback: bool = True) -> str:

  def f3(x: float) -> str:
    return f"{x:.3f}"

  html_table = (f"""<!-- HTML table for rich rendering -->
<table style="border-collapse:collapse; width:420px;">
  <thead>
    <tr>
      <th style="text-align:left; padding:6px 10px; border-bottom:1px solid #ddd;">Metric</th>
      <th style="text-align:right; padding:6px 10px; border-bottom:1px solid #ddd;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:6px 10px;">Best threshold (P(Not))</td><td style="text-align:right; padding:6px 10px;"><b>{f3(best_thr)}</b></td></tr>
    <tr><td style="padding:6px 10px;">Accuracy</td><td style="text-align:right; padding:6px 10px;">{f3(acc)}</td></tr>
    <tr><td style="padding:6px 10px;">Precision</td><td style="text-align:right; padding:6px 10px;">{f3(prec)}</td></tr>
    <tr><td style="padding:6px 10px;">Recall</td><td style="text-align:right; padding:6px 10px;">{f3(rec)}</td></tr>
    <tr><td style="padding:6px 10px;">F1</td><td style="text-align:right; padding:6px 10px;">{f3(f1)}</td></tr>
    <tr><td style="padding:6px 10px;">AUC (ROC)</td><td style="text-align:right; padding:6px 10px;">{f3(auc)}</td></tr>
    <tr><td style="padding:6px 10px;">Average Precision (PR)</td><td style="text-align:right; padding:6px 10px;">{f3(ap)}</td></tr>
  </tbody>
</table>""").strip()

  md_table = (f"""<!-- Markdown fallback table (for renderers that ignore HTML) -->
| **Metric**                | **Value** |
|:--------------------------|----------:|
| Best threshold *(P(Not))* | **{f3(best_thr)}** |
| Accuracy                  | {f3(acc)} |
| Precision                 | {f3(prec)} |
| Recall                    | {f3(rec)} |
| F1                        | {f3(f1)} |
| AUC (ROC)                 | {f3(auc)} |
| Average Precision (PR)    | {f3(ap)} |""").strip()

  body = (f"""### Model Performance

![PR Curve]({pr_curve_relpath})

{html_table}""").strip()

  if include_fallback:
    body += (f"""

<details>
<summary>Markdown table (fallback)</summary>

{md_table}
</details>""")

  return body


def _flush_left(s: str) -> str:
  # remove leading spaces/tabs on every line
  return "\n".join(line.lstrip() for line in s.splitlines()).strip()


def update_readme_performance(md_block: str,
                              repo_dir: Path,
                              start_marker: str = "<!-- PERF:START -->",
                              end_marker: str = "<!-- PERF:END -->") -> Path:
  readme = repo_dir / "README.md"
  readme.parent.mkdir(parents=True, exist_ok=True)

  block_clean = _flush_left(md_block)
  block = f"\n{start_marker}\n{block_clean}\n{end_marker}\n"

  text = readme.read_text(encoding="utf-8") if readme.exists() else "# vismod\n"
  pattern = re.compile(rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL)
  new_text = pattern.sub(block.strip("\n"), text) if pattern.search(text) else (text.rstrip("\n") +
                                                                                block)

  readme.write_text(new_text, encoding="utf-8")
  logger.info(f"Updated README: {readme}")
  return readme


# ---------- Evaluation ----------


@torch.no_grad()
def evaluate(checkpoint_path: str,
             data_dir: str,
             batch_size: int = 32,
             num_workers: int = 4,
             results_csv_fname: str = "eval_results.csv",
             make_report: bool = False,
             repo_dir: str = None):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  for entry in (Path.cwd() / "weights").iterdir():
    print(entry)

  logger.info(f"Device: {device}")
  logger.info(f"Loading model from checkpoint: {checkpoint_path}")
  model = ModClassifier.load_from_checkpoint(checkpoint_path)
  model.eval()
  model.to(device)

  logger.info(f"Loading test data from: {data_dir}/test/")
  data_module = ImageDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
  data_module.setup(stage="test")
  test_dataset = data_module.test_dataset
  test_loader = data_module.test_dataloader()

  acc = BinaryAccuracy().to(device)
  prec = BinaryPrecision().to(device)
  recall = BinaryRecall().to(device)
  f1 = BinaryF1Score().to(device)
  auc = BinaryAUROC().to(device)
  ap = BinaryAveragePrecision().to(device)

  all_probs, all_labels = [], []
  for batch in test_loader:
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x).squeeze(1).float()
    probs = torch.sigmoid(logits)

    all_probs.append(probs.detach().cpu())
    all_labels.append(y.detach().cpu())

  probs = torch.cat(all_probs)
  labels = torch.cat(all_labels)

  acc_val = acc(probs, labels).item()
  prec_val = prec(probs, labels).item()
  recall_val = recall(probs, labels).item()
  f1_val = f1(probs, labels).item()
  auc_val = auc(probs, labels).item()
  ap_val = ap(probs, labels).item()

  logger.info("\nEvaluation Metrics:")
  logger.info(f"Accuracy:  {acc_val:.4f}")
  logger.info(f"Precision: {prec_val:.4f}")
  logger.info(f"Recall:    {recall_val:.4f}")
  logger.info(f"F1 Score:  {f1_val:.4f}")
  logger.info(f"AUC:       {auc_val:.4f}")
  logger.info(f"AP:        {ap_val:.4f}")

  results_path = Path(results_csv_fname)
  row = {
      "checkpoint": str(checkpoint_path),
      "data_dir": str(data_dir),
      "accuracy": acc_val,
      "precision": prec_val,
      "recall": recall_val,
      "f1_score": f1_val,
      "auc": auc_val,
      "ap": ap_val,
  }
  df = pd.DataFrame([row])
  if results_path.exists():
    df.to_csv(results_path, mode="a", header=False, index=False)
  else:
    df.to_csv(results_path, index=False)

  # OPTIONAL: Generate report to get into the README.md
  # --------- Report: PR curve + README table (best-F1 threshold) ----------
  if not make_report:
    return

  logger.info("Making report")
  repo_dir = Path.cwd() if repo_dir == None else Path(repo_dir)
  assets_dir = repo_dir / "assets"
  assets_dir.mkdir(parents=True, exist_ok=True)
  pr_curve_path = assets_dir / "pr_curve.png"
  save_pr_curve(probs, labels, pr_curve_path, ap_val)

  # Best-F1 sweep for the README table
  ths = torch.linspace(0.01, 0.99, 99)
  f1_scores = []
  for t in ths:
    preds = (probs >= t).int()
    f1_scores.append(BinaryF1Score()(preds, labels).item())
  best_idx = int(torch.tensor(f1_scores).argmax().item())
  best_thr = float(ths[best_idx].item())

  preds_best = (probs >= best_thr).int()
  acc_b = BinaryAccuracy()(preds_best, labels).item()
  prec_b = BinaryPrecision()(preds_best, labels).item()
  rec_b = BinaryRecall()(preds_best, labels).item()
  f1_b = BinaryF1Score()(preds_best, labels).item()

  md = render_perf_section(
      best_thr=best_thr,
      acc=acc_b,
      prec=prec_b,
      rec=rec_b,
      f1=f1_b,
      auc=auc_val,
      ap=ap_val,
      pr_curve_relpath="assets/pr_curve.png",
  )

  update_readme_performance(md, repo_dir)


def main():
  parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
  parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
  parser.add_argument("--data_dir", required=True, help="Path to dataset root")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--log_file", type=str, default=None)
  parser.add_argument("--log_level", default="INFO")
  parser.add_argument("--results_csv_fname", type=str, default="eval_results.csv")
  parser.add_argument(
      "--make_report", action="store_true", help="If set, save PR curve and update README.md")
  parser.add_argument(
      "--repo_dir", type=str, default=".", help="Base of repo: contains README.md and assets/")

  args = parser.parse_args()
  setup_logging(args.log_file, args.log_level)
  seed_everything(0)
  evaluate(
      checkpoint_path=args.checkpoint,
      data_dir=args.data_dir,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      results_csv_fname=args.results_csv_fname,
      make_report=args.make_report,
      repo_dir=args.repo_dir)


if __name__ == "__main__":
  main()
