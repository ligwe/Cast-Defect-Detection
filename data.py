from typing import Tuple
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_t, eval_t, eval_t


def build_dataloaders(
    data_dir: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    train_t, val_t, test_t = build_transforms(image_size)

    train_path = os.path.join(data_dir, train_dir)
    val_path = os.path.join(data_dir, val_dir)
    test_path = os.path.join(data_dir, test_dir)

    train_ds = datasets.ImageFolder(train_path, transform=train_t)
    val_ds = datasets.ImageFolder(val_path, transform=val_t)
    test_ds = datasets.ImageFolder(test_path, transform=test_t)

    class_to_idx = train_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx
