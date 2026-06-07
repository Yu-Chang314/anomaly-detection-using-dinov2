from __future__ import annotations

import base64
import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from flask import Flask, jsonify, render_template, request
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import SSPTT


STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

CONFIG = {
    "img_size": 224,
    "patch_size": 14,
    "embed_dim": 1024,
    "num_heads": 10,
    "num_layers": 6,
    "mask_ratio": 0.1,
    "dropout": 0.0,
    "drop_path_rate": 0.0,
    "num_tokens": 16 * 16,
    "tokenizer_name": "dinov2_vitl14_reg",
    "repo_or_dir": "facebookresearch/dinov2",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "wood_ep300.pth"
CHECKPOINT_PATH = Path(os.getenv("SSPTT_CHECKPOINT", DEFAULT_CHECKPOINT))
THRESHOLD = float(os.getenv("SSPTT_THRESHOLD", "0.5"))

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATE_DIR),
)


class DINOWrapper(nn.Module):
    def __init__(self, repo_or_dir: Any, model: Any, **kwargs: Any) -> None:
        super().__init__()
        self.tokenizer = torch.hub.load(repo_or_dir, model, pretrained=True, **kwargs)
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, n: int = 1) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer.get_intermediate_layers(x, n=n)[0]
        return tokens


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def colorize_heatmap(scores: np.ndarray) -> Image.Image:
    scores = np.clip(scores, 0.0, 1.0)
    red = scores
    green = 1.0 - np.abs(scores - 0.5) * 2.0
    blue = 1.0 - scores
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def make_overlay(original: Image.Image, heatmap: Image.Image) -> Image.Image:
    original = original.convert("RGB").resize(heatmap.size, Image.Resampling.BILINEAR)
    return Image.blend(original, heatmap.convert("RGB"), alpha=0.45)


def make_mask(scores: np.ndarray) -> Image.Image:
    mask = (scores >= THRESHOLD).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def preprocess(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    clean_image = image.convert("RGB")
    transform = v2.Compose(
        [
            v2.Resize(CONFIG["img_size"], interpolation=v2.InterpolationMode.LANCZOS),
            v2.CenterCrop(CONFIG["img_size"]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )
    tensor = transform(clean_image).unsqueeze(0)
    preview = clean_image.resize(
        (CONFIG["img_size"], CONFIG["img_size"]), Image.Resampling.BILINEAR
    )
    return tensor, preview


@lru_cache(maxsize=1)
def load_model() -> tuple[SSPTT, torch.device]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Set SSPTT_CHECKPOINT to a trained .pth file."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSPTT(
        tokenizer=DINOWrapper(CONFIG["repo_or_dir"], CONFIG["tokenizer_name"]),
        dim=CONFIG["embed_dim"],
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_tokens"],
        mask_ratio=CONFIG["mask_ratio"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        drop_path_rate=CONFIG["drop_path_rate"],
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threshold": THRESHOLD,
    })


@app.post("/api/predict")
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"detail": "Please upload an image file."}), 400

    if not uploaded_file.content_type.startswith("image/"):
        return jsonify({"detail": "Please upload an image file."}), 400

    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
    except Exception:
        return jsonify({"detail": "Could not read image."}), 400

    try:
        model, device = load_model()
    except FileNotFoundError as error:
        return jsonify({"detail": str(error)}), 503

    tensor, preview = preprocess(image)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor, return_patch_level_masks=False)
        scores = F.softmax(logits[0], dim=0)[1].detach().cpu().numpy()

    scores = gaussian_filter(scores, sigma=4)
    score = float(np.max(scores))
    heatmap = colorize_heatmap(scores)
    mask = make_mask(scores)
    overlay = make_overlay(preview, heatmap)

    return jsonify({
        "filename": uploaded_file.filename,
        "score": score,
        "threshold": THRESHOLD,
        "label": "abnormal" if score >= THRESHOLD else "normal",
        "checkpoint": str(CHECKPOINT_PATH),
        "images": {
            "original": image_to_data_url(preview),
            "heatmap": image_to_data_url(heatmap),
            "overlay": image_to_data_url(overlay),
            "mask": image_to_data_url(mask.convert("RGB")),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
