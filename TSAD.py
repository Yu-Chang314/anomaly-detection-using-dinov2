import argparse
import gc
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import lightning as L

import torchvision.transforms.v2 as v2

from SSPTT import DINOWrapper, denormalize
from anomaly_types.cutpaste import CutPasteNormal, CutPasteScar
from anomaly_types.nsa import NSAAnomalyGenerator, TEXTURES
from anomaly_types.perlin import PerlinAnomalyGenerator
from mvtec import MVTecDataset
from teacher_student import StudentNet, TeacherNet

warnings.filterwarnings("ignore")

# ==========================================
# 0. Global Configuration
# ==========================================
USE_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_CUDA else "cpu"
CONFIG = {
    "seed": 42,
    "img_size": 224,
    "patch_size": 14,  # DINOv2 ViT-L/14 -> 16x16 token grid
    "class_name": "grid",
    "embed_dim": 1024,
    "tokenizer_name": "dinov2_vitl14_reg",
    "repo_or_dir": "facebookresearch/dinov2",
    # teacher distillation
    "teacher_width": 512,
    "teacher_epochs": 200,
    "teacher_lr": 1e-4,
    "teacher_batch_size": 8,
    # student training (sized for an 8GB GPU, e.g. RTX 5060 laptop)
    "student_cnn_dim": 512,
    "student_swin_dim": 128,  # C: stage dims C/2C/4C/8C
    "student_swin_depth": 4,  # Swin blocks per stage (MSTUnet ablation: 4)
    "student_epochs": 300,
    "student_lr": 1e-4,
    "student_batch_size": 8,
    "warmup_epochs": 5,
    "margin": 1.0,  # target cosine distance on defect patches
    "lambda_anomaly": 1.0,
    # image score = mean of the top-k anomaly-map pixels (~2% of 224x224);
    # plain max is destroyed by single false-positive patches on normal images
    "score_top_k": 1000,
    "anomaly_probability": 1.0,  # chance of injecting a synthetic defect
    "checkpoint_dir": "./checkpoints",
    "num_workers": 4,
    "precision": "bf16-mixed",
}

DATA_ROOT = "./dataset/mvtec_anomaly_detection"
DTD_ROOT = "./datasets/dtd/dtd/images"

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def seed_numpy_worker(_: int) -> None:
    # NSA's patch_ex uses numpy RNG, which is not reseeded per worker by default
    np.random.seed(torch.utils.data.get_worker_info().seed % 2**32)


def train_augment(class_name: str) -> v2.Transform | None:
    """Geometric augmentation of normal training images. Textures have natural
    variation that the fixed resize+center-crop pipeline never shows the
    student, causing global divergence on slightly different test normals."""
    if class_name in TEXTURES:
        return v2.Compose([v2.RandomVerticalFlip(), v2.RandomHorizontalFlip()])
    return None


# ==========================================
# 1. Teacher distillation (fit DINOv2 features)
# ==========================================
class TeacherDistillLightning(L.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = TeacherNet(
            out_dim=config["embed_dim"], width=config["teacher_width"]
        )
        self.dino = DINOWrapper(config["repo_or_dir"], config["tokenizer_name"])
        self.grid_size = config["img_size"] // config["patch_size"]

    def dino_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.dino(x)  # [B, N, D]
        b, _, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)

    def training_step(self, batch, _) -> torch.Tensor:
        clean_images, _, _, _ = batch
        target = self.dino_feature_map(clean_images)
        pred = self.model(clean_images)
        loss = F.mse_loss(pred, target)
        self.log("distill loss", loss, prog_bar=True, batch_size=clean_images.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.config["teacher_lr"], weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config["teacher_epochs"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ==========================================
# 2. Student training (mimic teacher on normal, diverge on defects)
# ==========================================
class StudentLightning(L.LightningModule):
    def __init__(self, config: dict, teacher: TeacherNet, output_path: str) -> None:
        super().__init__()
        self.config = config
        self.output_path = output_path
        self.model = StudentNet(
            out_dim=config["embed_dim"],
            cnn_dim=config["student_cnn_dim"],
            swin_embed_dim=config["student_swin_dim"],
            swin_depth=config["student_swin_depth"],
        )
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def feature_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch cosine distance between teacher and student features. [B, H, W]"""
        with torch.no_grad():
            t = F.normalize(self.teacher(x), dim=1)
        s = F.normalize(self.model(x), dim=1)
        return 1.0 - (t * s).sum(dim=1)

    def training_step(self, batch, _) -> torch.Tensor:
        _, augmented_images, masks, _ = batch

        patch_mask = (
            F.max_pool2d(
                masks,
                kernel_size=self.config["patch_size"],
                stride=self.config["patch_size"],
            ).squeeze(1)
            > 0.5
        )

        distance = self.feature_distance(augmented_images)

        normal_d = distance[~patch_mask]
        anomaly_d = distance[patch_mask]

        loss_normal = (
            normal_d.mean() if normal_d.numel() > 0 else distance.sum() * 0.0
        )
        loss_anomaly = (
            F.relu(self.config["margin"] - anomaly_d).mean()
            if anomaly_d.numel() > 0
            else distance.sum() * 0.0
        )
        loss = loss_normal + self.config["lambda_anomaly"] * loss_anomaly

        bs = augmented_images.size(0)
        self.log("loss", loss, prog_bar=True, batch_size=bs)
        self.log("normal loss", loss_normal, prog_bar=True, batch_size=bs)
        self.log("anomaly loss", loss_anomaly, prog_bar=True, batch_size=bs)
        return loss

    def on_predict_start(self) -> None:
        self.image_scores: list[float] = []
        self.image_labels: list[int] = []
        self.pixel_scores_all: list[float] = []
        self.pixel_labels_all: list[int] = []
        os.makedirs(self.output_path, exist_ok=True)

    def predict_step(self, batch, batch_idx: int):
        images, labels, masks, paths = batch
        distance = self.feature_distance(images)  # [B, 16, 16]
        anomaly_maps = F.interpolate(
            distance.unsqueeze(1), size=images.shape[2:], mode="bilinear"
        ).squeeze(1)
        threshold = 0.5

        for sample_idx in range(images.size(0)):
            amap = anomaly_maps[sample_idx].float().detach().cpu().numpy()
            amap = gaussian_filter(amap, sigma=4)

            top_k = self.config["score_top_k"]
            score = float(np.sort(amap.flatten())[-top_k:].mean())
            try:
                image_label = int(labels[sample_idx].item())
            except Exception:
                image_label = int(labels[sample_idx])

            self.image_scores.append(score)
            self.image_labels.append(image_label)

            mask_np = masks[sample_idx].squeeze().detach().cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            self.pixel_scores_all.extend(amap.flatten().tolist())
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
            plt.imshow(amap, cmap="jet", vmin=0, vmax=1)
            plt.title(f"Heatmap (Score: {score:.3f})")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(mask_np, cmap="gray")
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(amap > threshold, cmap="gray")
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
            self.model.parameters(), lr=self.config["student_lr"], weight_decay=1e-4
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, total_iters=self.config["warmup_epochs"]),
                CosineAnnealingLR(
                    optimizer,
                    T_max=self.config["student_epochs"] - self.config["warmup_epochs"],
                ),
            ],
            milestones=[self.config["warmup_epochs"]],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ==========================================
# 3. Training phases
# ==========================================
def make_trainer(config: dict, max_epochs: int, limit_batches=None) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator=ACCELERATOR,
        devices=1,
        log_every_n_steps=1,
        precision=config["precision"] if USE_CUDA else "32-true",
        benchmark=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=limit_batches,
        limit_predict_batches=limit_batches,
    )


def make_loader(config: dict, dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
        worker_init_fn=seed_numpy_worker,
    )


def distill_teacher(config: dict) -> TeacherNet:
    class_name = config["class_name"]
    ckpt_path = os.path.join(
        config["checkpoint_dir"], f"tsad_teacher_{class_name}.pth"
    )

    if os.path.exists(ckpt_path):
        print(f"[TEACHER] Loading cached teacher from {ckpt_path}")
        teacher = TeacherNet(
            out_dim=config["embed_dim"], width=config["teacher_width"]
        )
        teacher.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return teacher

    print(f"\n{'=' * 50}\n  Phase 1: distilling teacher ({class_name})\n{'=' * 50}")
    train_ds = MVTecDataset(
        DATA_ROOT, class_name, phase="train", augment=train_augment(class_name)
    )
    train_loader = make_loader(
        config, train_ds, config["teacher_batch_size"], shuffle=True
    )

    lit_module = TeacherDistillLightning(config)
    trainer = make_trainer(
        config, config["teacher_epochs"], config.get("limit_batches")
    )
    trainer.fit(lit_module, train_dataloaders=train_loader)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    torch.save(lit_module.model.state_dict(), ckpt_path)
    print(f"[TEACHER] Saved to {ckpt_path}")

    # Free DINOv2, the trainer, and the dataloader workers before the student
    # phase spawns its own workers; the brief overlap can exhaust system RAM.
    teacher = lit_module.model
    del lit_module, trainer, train_loader
    gc.collect()
    if USE_CUDA:
        torch.cuda.empty_cache()
    return teacher


def train_student(config: dict, teacher: TeacherNet, output_path: str) -> StudentLightning:
    class_name = config["class_name"]
    p = config["anomaly_probability"]

    train_ds = MVTecDataset(
        DATA_ROOT,
        class_name,
        phase="train",
        augment=train_augment(class_name),
        anomaly_generators=[
            NSAAnomalyGenerator(
                class_name,
                source_dir=os.path.join(DATA_ROOT, class_name, "train", "good"),
                probability=p,
            ),
            PerlinAnomalyGenerator(
                anomaly_source_path=DTD_ROOT,
                probability=p,
                blend_factor=(0.1, 1.0),
            ),
            CutPasteNormal(probability=p),
            CutPasteScar(probability=p, length_range=(10, 224)),
        ],
    )
    train_loader = make_loader(
        config, train_ds, config["student_batch_size"], shuffle=True
    )

    print(f"\n{'=' * 50}\n  Phase 2: training student ({class_name})\n{'=' * 50}")
    lit_module = StudentLightning(config, teacher, output_path=output_path)
    trainer = make_trainer(
        config, config["student_epochs"], config.get("limit_batches")
    )
    trainer.fit(lit_module, train_dataloaders=train_loader)

    ckpt_path = os.path.join(
        config["checkpoint_dir"], f"tsad_student_{class_name}.pth"
    )
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    torch.save(lit_module.model.state_dict(), ckpt_path)
    print(f"[STUDENT] Saved to {ckpt_path}")
    return lit_module


def evaluate(config: dict, lit_module: StudentLightning) -> None:
    class_name = config["class_name"]
    print(f"\n[EVAL] {class_name} — Starting Inference...")
    test_ds = MVTecDataset(DATA_ROOT, class_name, phase="test")
    test_loader = make_loader(config, test_ds, batch_size=1, shuffle=False)
    trainer = make_trainer(config, 1, config.get("limit_batches"))
    trainer.predict(lit_module, dataloaders=test_loader)


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", default=CONFIG["class_name"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch run on a few batches to check the pipeline end-to-end",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="skip training and evaluate cached teacher/student checkpoints",
    )
    args = parser.parse_args()

    config = {**CONFIG, "class_name": args.class_name}
    if args.smoke:
        config.update(
            {
                "teacher_epochs": 1,
                "student_epochs": 1,
                "warmup_epochs": 0,
                "limit_batches": 4,
                "num_workers": 0,
            }
        )

    L.seed_everything(config["seed"], workers=True)

    output_path = f"./results/{config['class_name']}_tsad"
    os.makedirs(output_path, exist_ok=True)

    teacher = distill_teacher(config)
    if args.eval_only:
        student_ckpt = os.path.join(
            config["checkpoint_dir"], f"tsad_student_{config['class_name']}.pth"
        )
        student_module = StudentLightning(config, teacher, output_path=output_path)
        student_module.model.load_state_dict(
            torch.load(student_ckpt, weights_only=True)
        )
    else:
        student_module = train_student(config, teacher, output_path)
    evaluate(config, student_module)
