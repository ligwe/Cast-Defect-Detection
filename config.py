import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    data_dir: str = "data"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    output_dir: str = "outputs"

    model_name: str = "vit_base_patch16_224"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 4

    seed: int = 42
    mixed_precision: bool = True
    freeze_backbone: bool = False

    num_classes: int = 2


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ViT classifier for car defect detection")
    parser.add_argument("--data_dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--train_dir", type=str, default=TrainConfig.train_dir)
    parser.add_argument("--val_dir", type=str, default=TrainConfig.val_dir)
    parser.add_argument("--test_dir", type=str, default=TrainConfig.test_dir)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)

    parser.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--image_size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)

    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--mixed_precision", action="store_true", default=TrainConfig.mixed_precision)
    parser.add_argument("--no_mixed_precision", action="store_false", dest="mixed_precision")
    parser.add_argument("--freeze_backbone", action="store_true", default=TrainConfig.freeze_backbone)

    parser.add_argument("--num_classes", type=int, default=TrainConfig.num_classes)

    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        freeze_backbone=args.freeze_backbone,
        num_classes=args.num_classes,
    )
