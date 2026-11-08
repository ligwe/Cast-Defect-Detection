import os
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import parse_args
from src.data import build_dataloaders
from src.model import create_model
from src.utils.common import load_checkpoint, get_device


def evaluate_test(model: nn.Module, loader, device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    report = classification_report(all_targets, all_preds, target_names=["non_defective", "defective"], digits=4)
    cm = confusion_matrix(all_targets, all_preds)
    print(report)
    print("Confusion matrix:\n", cm)
    return {}


def main() -> None:
    cfg = parse_args()
    device = get_device()
    _, _, test_loader, class_to_idx = build_dataloaders(
        cfg.data_dir, cfg.train_dir, cfg.val_dir, cfg.test_dir,
        cfg.image_size, cfg.batch_size, cfg.num_workers
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = create_model(cfg.model_name, cfg.num_classes, cfg.freeze_backbone)
    ckpt_path = os.path.join(cfg.output_dir, "best.pt")
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"]) 
    model.to(device)

    evaluate_test(model, test_loader, device)


if __name__ == "__main__":
    main()
