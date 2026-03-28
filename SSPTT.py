import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
from typing import Optional, Any
from scipy.ndimage import gaussian_filter
from anomaly_types.perlin import PerlinAnomalyGenerator
from anomaly_types.cutpaste import CutPasteNormal, CutPasteScar
from mvtec import MVTecDataset
import lightning as L
from model import SSPTT

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

# ==========================================
# 0. Global Configuration (Strictly aligned with Paper)
# ==========================================
USE_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_CUDA else "cpu"
DEVICE = "cuda" if USE_CUDA else "cpu"
CONFIG = {
    "seed": 42,
    "img_size": 224,  # Paper IV.A
    "patch_size": 14,  # DINOv2(14) / DINOv3(16)
    "class_name": "carpet",  # Change this to the desired class
    "embed_dim": 1024,  # ViT-B
    "num_heads": 10,
    "num_layers": 6,
    "num_landmarks": 64,  # Paper III.C
    "epochs": 300,  # Paper IV.B
    "warmup_epochs": 5,
    "batch_size": 4,  # Paper IV.B
    "lr": 5e-5,  # Paper IV.B (Important: 1e-5 is small, requires stable training)
    "mask_ratio": 0.1,  # Paper III.B
    "lambda_mse": 1.0,  # Paper Eq 6
    "device": DEVICE,
    "checkpoint_dir": "./checkpoints",
    "dropout": 0.0,  # Added dropout for stability
    "drop_path_rate": 0.0,  # Added stochastic depth for stability
    "num_tokens": 16 * 16,  # (img_size // patch_size) ** 2
    "tokenizer_name": "dinov2_vitl14_reg",
    "repo_or_dir": "facebookresearch/dinov2",
    "num_workers": 4,
    "precision": "bf16-mixed",
}

# CONFIG = {
#     "seed": 42,
#     "img_size": 224,  # Paper IV.A
#     "patch_size": 16,  # DINOv2(14) / DINOv3(16)
#     "class_name": "cable",  # Change this to the desired class
#     "embed_dim": 1024,  # ViT-B
#     "num_heads": 10,
#     "num_layers": 6,
#     "num_landmarks": 64,  # Paper III.C
#     "epochs": 300,  # Paper IV.B
#     "warmup_epochs": 10,
#     "batch_size": 4,  # Paper IV.B
#     "lr": 5e-5,  # Paper IV.B (Important: 1e-5 is small, requires stable training)
#     "mask_ratio": 0.1,  # Paper III.B
#     "lambda_mse": 1.0,  # Paper Eq 6
#     "device": DEVICE,
#     "checkpoint_dir": "./checkpoints",
#     "dropout": 0.1,  # Added dropout for stability
#     "drop_path_rate": 0.0,  # Added stochastic depth for stability
#     "num_tokens": 14 * 14,  # (img_size // patch_size) ** 2
#     "tokenizer_name": "dinov3_vitl16",
#     "repo_or_dir": "facebookresearch/dinov3",
#     "num_workers": 4,
#     "precision": "bf16-mixed",
# }


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # For better numerical stability
torch.use_deterministic_algorithms(False)


def denormalize(
    tensor: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


class DINOWrapper(nn.Module):
    def __init__(self, repo_or_dir: Any, model: Any, **kargs) -> None:
        super().__init__()
        self.tokenizer = torch.hub.load(repo_or_dir, model, pretrained=True, **kargs)
        self.tokenizer.eval()
        for pram in self.tokenizer.parameters():
            pram.requires_grad = False

    def forward(self, x: torch.Tensor, n: int = 1) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer.get_intermediate_layers(x, n=n)[0]
        return tokens


class SSPTTLightning(L.LightningModule):
    def __init__(self, config: dict, output_path: str) -> None:
        super().__init__()
        self.config = config
        self.output_path = output_path
        self.model = SSPTT(
            tokenizer=DINOWrapper(config["repo_or_dir"], config["tokenizer_name"]),
            dim=config["embed_dim"],
            patch_size=config["patch_size"],
            num_patches=config["num_tokens"],
            mask_ratio=config["mask_ratio"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            drop_path_rate=config["drop_path_rate"],
        )
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        clean_x: Optional[torch.Tensor] = None,
        return_patch_level_masks: bool = True,
    ) -> torch.Tensor:
        return self.model(
            x, clean_x=clean_x, return_patch_level_masks=return_patch_level_masks
        )

    def training_step(self, batch, _) -> torch.Tensor:
        clean_images, augmented_images, masks, _ = batch

        patch_mask = F.max_pool2d(
            masks,
            kernel_size=self.config["patch_size"],
            stride=self.config["patch_size"],
        )

        pred_masks, recon_tokens, clean_tokens = self(
            augmented_images, clean_x=clean_images
        )

        loss_mse = self.mse_loss_fn(recon_tokens, clean_tokens)
        loss_ce = self.ce_loss_fn(pred_masks, patch_mask.long().squeeze(1))
        loss = loss_ce + self.config["lambda_mse"] * loss_mse

        bs = clean_images.size(0)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=bs,
        )
        self.log(
            "mse loss",
            loss_mse,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=bs,
        )
        self.log(
            "ce loss",
            loss_ce,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=bs,
        )
        return loss

    def on_predict_start(self) -> None:
        self.image_scores: list[float] = []
        self.image_labels: list[int] = []
        self.pixel_scores_all: list[float] = []
        self.pixel_labels_all: list[int] = []
        os.makedirs(self.output_path, exist_ok=True)

    def predict_step(self, batch, batch_idx: int):
        images, labels, masks, paths = batch
        logits = self(images, return_patch_level_masks=False)
        threshold = 0.5

        for sample_idx in range(images.size(0)):
            probs = F.softmax(logits[sample_idx], dim=0)[1].detach().cpu().numpy()
            probs = gaussian_filter(probs, sigma=4)

            patch_score = float(np.max(probs))
            try:
                image_label = int(labels[sample_idx].item())
            except Exception:
                image_label = int(labels[sample_idx])

            self.image_scores.append(patch_score)
            self.image_labels.append(image_label)

            mask_np = masks[sample_idx].squeeze().detach().cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            self.pixel_scores_all.extend(probs.flatten().tolist())
            self.pixel_labels_all.extend(mask_bin.flatten().tolist())

            path = paths[sample_idx]
            defect_type = os.path.basename(os.path.dirname(path))
            plt.figure(figsize=(20, 5))

            plt.subplot(1, 4, 1)
            plt.imshow(
                np.clip(
                    denormalize(images[sample_idx])
                    .detach()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy(),
                    0,
                    1,
                )
            )
            plt.title(f"{defect_type}")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(probs, cmap="jet", vmin=0, vmax=1)
            plt.title(f"Heatmap (Max: {patch_score:.3f})")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(mask_np, cmap="gray")
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(probs > threshold, cmap="gray")
            plt.title("Predicted")
            plt.axis("off")

            plt.savefig(
                os.path.join(
                    self.output_path, f"{defect_type}_{batch_idx + sample_idx:03d}.png"
                )
            )
            plt.close()

    def on_predict_epoch_end(self) -> None:
        try:
            i_auroc = roc_auc_score(self.image_labels, self.image_scores)
            p_auroc = roc_auc_score(self.pixel_labels_all, self.pixel_scores_all)
        except Exception as error:
            warnings.warn(f"AUROC calculation failed: {error}")
            i_auroc = float("nan")
            p_auroc = float("nan")

        print(f"I-AUROC: {i_auroc}")
        print(f"P-AUROC: {p_auroc}")

        with open(os.path.join(self.output_path, "metrics.txt"), "w") as f:
            f.write(f"I-AUROC: {i_auroc}\n")
            f.write(f"P-AUROC: {p_auroc}\n")

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["lr"],
            weight_decay=1e-4,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, total_iters=self.config["warmup_epochs"]),
                CosineAnnealingLR(
                    optimizer,
                    T_max=self.config["epochs"] - self.config["warmup_epochs"],
                ),
            ],
            milestones=[self.config["warmup_epochs"]],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class SavePTHCallback(L.Callback):
    def __init__(
        self, class_name: str, checkpoint_dir: str, every_n_epochs: int = 50
    ) -> None:
        super().__init__()
        self.class_name = class_name
        self.checkpoint_dir = checkpoint_dir
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: SSPTTLightning) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0:
            torch.save(
                pl_module.model.state_dict(),
                os.path.join(self.checkpoint_dir, f"{self.class_name}_ep{epoch}.pth"),
            )


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    L.seed_everything(CONFIG["seed"], workers=True)

    DATA_ROOT = "./dataset/mvtec_anomaly_detection"
    DTD_ROOT = "./datasets/dtd/dtd/images"
    CLASS_NAME = CONFIG["class_name"]
    OUTPUT_PATH = f"./results/{CLASS_NAME}_v22"
    CKPT_DIR = CONFIG["checkpoint_dir"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = CONFIG["device"]

    lit_module = SSPTTLightning(CONFIG, output_path=OUTPUT_PATH)

    if os.path.exists(os.path.join(DATA_ROOT, CLASS_NAME)):
        train_ds = MVTecDataset(
            DATA_ROOT,
            CLASS_NAME,
            phase="train",
            anomaly_generators=[
                PerlinAnomalyGenerator(
                    anomaly_source_path=DTD_ROOT,
                    probability=1.0,
                    blend_factor=(0.1, 1.0),
                ),
                CutPasteNormal(
                    probability=1.0,
                ),
                CutPasteScar(
                    probability=1.0,
                    # width_range=(2, 10),
                    length_range=(10, 224),
                ),
            ],
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
            persistent_workers=CONFIG["num_workers"] > 0,
            prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
        )
        print(f"Training {CLASS_NAME} with Lightning...")
        trainer = L.Trainer(
            max_epochs=CONFIG["epochs"],
            accelerator=ACCELERATOR,
            devices=1,
            deterministic=False,
            log_every_n_steps=1,
            precision=CONFIG["precision"] if USE_CUDA else "32-true",
            benchmark=True,
            enable_model_summary=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[SavePTHCallback(CLASS_NAME, CKPT_DIR, every_n_epochs=50)],
        )
        trainer.fit(lit_module, train_dataloaders=train_loader)

    print("\n[INIT] Starting Inference...")
    test_ds = MVTecDataset(DATA_ROOT, CLASS_NAME, phase="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
    )
    trainer.predict(lit_module, dataloaders=test_loader)

    print(f"Done. Check results in {OUTPUT_PATH}")
