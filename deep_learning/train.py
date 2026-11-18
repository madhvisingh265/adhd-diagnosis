"""
Train and evaluate EEGNet+LSTM with SUBJECT-WISE cross-validation.

Why subject-wise: each child contributes hundreds of overlapping windows. If windows from the same
child land in both train and test, the model can recognise the *child* rather than ADHD, and
accuracy looks far better than it really is (a common mistake in published EEG work). Here every
test fold contains only children the model has never seen.

Two scores are reported:
  - window-level : each 2-second window classified on its own
  - child-level  : average the window probabilities per child -> one decision per child
                   (this is the number that matters for a screening tool)

Usage:
    python src/train.py --folds 5 --epochs 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from data import load_windows
from model import EEGNetLSTM

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(42)
np.random.seed(42)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")}


def train_fold(Xtr, ytr, Xva, yva, epochs, device):
    model = EEGNetLSTM(n_channels=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            # light augmentation: random amplitude scaling + gaussian noise
            xb = xb * (1 + 0.1 * torch.randn(xb.size(0), 1, 1, device=device)) + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(model(torch.from_numpy(Xva[i:i + 256]).to(device)), 1)[:, 1].cpu()
                       for i in range(0, len(Xva), 256)]).numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, groups = load_windows()
    print(f"{X.shape[0]} windows from {len(np.unique(groups))} children on {device}")
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_win_y, all_win_p, child_rows, fold_scores = [], [], [], []
    for k, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        p = train_fold(X[tr], y[tr], X[va], y[va], args.epochs, device)
        all_win_y.append(y[va]); all_win_p.append(p)
        fold_child = {}
        for g in np.unique(groups[va]):
            m = groups[va] == g
            fold_child[g] = (y[va][m][0], p[m].mean())
        cy = np.array([v[0] for v in fold_child.values()]); cp = np.array([v[1] for v in fold_child.values()])
        child_rows += list(zip(fold_child.keys(), cy, cp))
        s = {"fold": k, "window": metrics(y[va], p), "child": metrics(cy, cp)}
        fold_scores.append(s)
        print(f"fold {k}: window acc {s['window']['accuracy']:.3f} | child acc {s['child']['accuracy']:.3f} "
              f"({len(cy)} children)")

    wy, wp = np.concatenate(all_win_y), np.concatenate(all_win_p)
    cy = np.array([r[1] for r in child_rows]); cp = np.array([r[2] for r in child_rows])
    summary = {"window_level": metrics(wy, wp), "child_level": metrics(cy, cp),
               "child_confusion_matrix": confusion_matrix(cy, (cp >= 0.5).astype(int)).tolist(),
               "folds": fold_scores, "n_windows": int(len(X)), "n_children": int(len(cy)),
               "epochs": args.epochs, "cv_folds": args.folds}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ["window_level", "child_level", "child_confusion_matrix"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
