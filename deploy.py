import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File
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


def create_app(ckpt_path: str, model_name: str = "vit_base_patch16_224", num_classes: int = 2, image_size: int = 224) -> FastAPI:
    device = get_device()
    model = load_model(ckpt_path, model_name, num_classes, device)
    t = build_infer_transform(image_size)

    app = FastAPI(title="Car Defect Classifier API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        conf, pred = torch.max(probs, dim=0)
        label = "defective" if int(pred.item()) == 1 else "non_defective"
        return {"label": label, "confidence": float(conf.item())}

    return app


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.ckpt, args.model_name, args.num_classes, args.image_size)
    uvicorn.run(app, host=args.host, port=args.port)
