import os
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.model import create_model
from src.utils.common import load_checkpoint, get_device


def build_infer_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path: str, model_name: str, num_classes: int, device) -> torch.nn.Module:
    model = create_model(model_name, num_classes)
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"]) 
    model.to(device)
    model.eval()
    return model


def infer_images(model, image_paths: List[str], image_size: int, device) -> List[Tuple[str, str, float]]:
    t = build_infer_transform(image_size)
    results = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            x = t(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
            conf, pred = torch.max(probs, dim=0)
            label = "defective" if int(pred.item()) == 1 else "non_defective"
            results.append((path, label, float(conf.item())))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on image(s)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--input", type=str, required=True, help="Path to image file or directory")

    args = parser.parse_args()

    device = get_device()

    if os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [args.input]

    model = load_model(args.ckpt, args.model_name, args.num_classes, device)
    results = infer_images(model, image_paths, args.image_size, device)

    for path, label, conf in results:
        print(f"{path}: {label} ({conf:.3f})")
