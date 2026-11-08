from typing import Tuple

import torch
from sklearn.metrics import precision_recall_fscore_support


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / max(total, 1)


def prf1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
    return float(precision), float(recall), float(f1)
