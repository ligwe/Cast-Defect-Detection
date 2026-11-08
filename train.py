import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import parse_args
from src.data import build_dataloaders
from src.model import create_model
from src.utils.common import set_seed, get_device, save_checkpoint, format_metrics
from src.utils.metrics import accuracy_from_logits, prf1_from_logits


def train_one_epoch(model: nn.Module, loader, optimizer, scaler, device) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)

    num_batches = len(loader)
    return {
        "loss": running_loss / max(num_batches, 1),
        "acc": running_acc / max(num_batches, 1),
    }


def validate(model: nn.Module, loader, device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, targets)
            p, r, f1 = prf1_from_logits(logits, targets)
            total_precision += p
            total_recall += r
            total_f1 += f1

    num_batches = len(loader)
    return {
        "loss": total_loss / max(num_batches, 1),
        "acc": total_acc / max(num_batches, 1),
        "precision": total_precision / max(num_batches, 1),
        "recall": total_recall / max(num_batches, 1),
        "f1": total_f1 / max(num_batches, 1),
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = get_device()
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        cfg.data_dir, cfg.train_dir, cfg.val_dir, cfg.test_dir,
        cfg.image_size, cfg.batch_size, cfg.num_workers
    )

    model = create_model(cfg.model_name, cfg.num_classes, cfg.freeze_backbone)
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    best_val_f1 = -1.0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}/{cfg.epochs} | Train: {format_metrics(train_metrics)} | Val: {format_metrics(val_metrics)}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "class_to_idx": class_to_idx,
                "config": vars(cfg),
                "val_metrics": val_metrics,
            }, cfg.output_dir, filename="best.pt")

    # Save final
    save_checkpoint({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": cfg.epochs,
        "class_to_idx": class_to_idx,
        "config": vars(cfg),
    }, cfg.output_dir, filename="last.pt")


if __name__ == "__main__":
    main()
