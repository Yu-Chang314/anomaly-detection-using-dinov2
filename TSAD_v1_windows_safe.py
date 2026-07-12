import argparse
import gc
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    "teacher_batch_size": 4,
    # student training (sized for an 8GB GPU, e.g. RTX 5060 laptop)
    "student_cnn_dim": 512,
    "student_swin_dim": 128,  # C: stage dims C/2C/4C/8C
    "student_swin_depth": 4,  # Swin blocks per stage (MSTUnet ablation: 4)
    "student_epochs": 300,
    "student_lr": 1e-4,
    "student_batch_size": 4,
    "warmup_epochs": 5,
    
    #"margin": 1.0,  # target cosine distance on defect patches
    #"lambda_anomaly": 1.0,
    
    # 原本的 "lambda_anomaly": 1.0 可以刪掉。
    # alpha_normal：正常區域 Teacher–Student 應相似
    # beta_anomaly：人工瑕疵區域 Teacher–Student 應分離
    # Student loss weights
    "alpha_normal": 2.0,
    "beta_anomaly": 0.5,
    "margin": 0.5,

    # 只使用最難的部分正常 patch 計算 normal loss
    # 1.0 = 全部正常 patch；0.10 = 最高距離的 10%
    "hard_normal_ratio": 1.0,

    # 使用正常訓練資料校正判斷門檻
    "pixel_threshold_quantile": 0.995,
    "image_threshold_quantile": 0.99,

    # image score = mean of the top-k anomaly-map pixels (~2% of 224x224);
    # plain max is destroyed by single false-positive patches on normal images
    "score_top_k": 1000,
    "anomaly_probability": 0.7,  # chance of injecting a synthetic defect
    "checkpoint_dir": "./checkpoints/calibrated_ts",
    "num_workers": 0,
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

    #def feature_distance(self, x: torch.Tensor) -> torch.Tensor:
    #    """Per-patch cosine distance between teacher and student features. [B, H, W]"""
    #    with torch.no_grad():
    #        t = F.normalize(self.teacher(x), dim=1)
    #    s = F.normalize(self.model(x), dim=1)
    #    return 1.0 - (t * s).sum(dim=1)

    def feature_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算每個 patch 的 Teacher–Student cosine distance。

        Teacher 已凍結，因此放入 no_grad。
        Student 必須保留 autograd，否則模型不會更新。
        """
        with torch.no_grad():
            teacher_features = self.teacher(x)
            teacher_features = F.normalize(teacher_features, dim=1)

        student_features = self.model(x)
        student_features = F.normalize(student_features, dim=1)

        distance = 1.0 - (teacher_features * student_features).sum(dim=1)

        return distance

    """
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
    """

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, augmented_images, masks, _ = batch

        # 將 224×224 pixel mask 縮小成 16×16 patch mask
        patch_mask = F.max_pool2d(
            masks,
            kernel_size=self.config["patch_size"],
            stride=self.config["patch_size"],
        ).squeeze(1) > 0.5

        # [B, 16, 16]
        distance = self.feature_distance(augmented_images)

        normal_distances = distance[~patch_mask]
        anomaly_distances = distance[patch_mask]

        # ---------------------------------------------------------
        # 1. Normal loss：只取最容易誤判的 hard normal patches
        # ---------------------------------------------------------
        if normal_distances.numel() > 0:
            hard_ratio = float(self.config["hard_normal_ratio"])
            hard_count = max(
                1,
                int(normal_distances.numel() * hard_ratio),
            )

            hard_normal_distances = torch.topk(
                normal_distances,
                k=hard_count,
                largest=True,
            ).values

            loss_normal = hard_normal_distances.mean()
        else:
            loss_normal = distance.sum() * 0.0

        # ---------------------------------------------------------
        # 2. Anomaly loss：異常 patch 距離必須至少達到 margin
        # ---------------------------------------------------------
        if anomaly_distances.numel() > 0:
            loss_anomaly = F.relu(
                self.config["margin"] - anomaly_distances
            ).mean()
        else:
            loss_anomaly = distance.sum() * 0.0

        alpha = float(self.config["alpha_normal"])
        beta = float(self.config["beta_anomaly"])

        weighted_normal = alpha * loss_normal
        weighted_anomaly = beta * loss_anomaly

        loss = weighted_normal + weighted_anomaly

        batch_size = augmented_images.size(0)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "loss_normal",
            loss_normal,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "loss_anomaly",
            loss_anomaly,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "weighted_normal",
            weighted_normal,
            batch_size=batch_size,
        )
        self.log(
            "weighted_anomaly",
            weighted_anomaly,
            batch_size=batch_size,
        )
        self.log(
            "mean_distance",
            distance.mean(),
            batch_size=batch_size,
        )

        return loss
    
    def on_after_backward(self) -> None:
        """
        確認 Student 確實有梯度。

        若 grad_norm 長期為 0，代表 Student 沒有真正參與反向傳播。
        """
        if self.global_step % 20 != 0:
            return

        grad_squared_sum = 0.0
        parameters_with_grad = 0

        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue

            grad_squared_sum += parameter.grad.detach().norm(2).item() ** 2
            parameters_with_grad += 1

        grad_norm = grad_squared_sum ** 0.5

        self.log(
            "student_grad_norm",
            grad_norm,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        if parameters_with_grad == 0:
            raise RuntimeError(
                "Student 沒有任何梯度。"
                "請檢查 Student forward 是否被放進 torch.no_grad()。"
            )

    def on_predict_start(self) -> None:
        self.image_scores: list[float] = []
        self.image_labels: list[int] = []
        self.image_predictions: list[int] = []

        self.pixel_scores_all: list[float] = []
        self.pixel_labels_all: list[int] = []
        self.pixel_predictions_all: list[int] = []

        if not hasattr(self, "pixel_threshold"):
            raise RuntimeError(
                "尚未設定 pixel_threshold，請先執行 calibrate_thresholds()。"
            )

        if not hasattr(self, "image_threshold"):
            raise RuntimeError(
                "尚未設定 image_threshold，請先執行 calibrate_thresholds()。"
            )

        os.makedirs(self.output_path, exist_ok=True)

    def predict_step(self, batch, batch_idx: int):
        images, labels, masks, paths = batch
        distance = self.feature_distance(images)  # [B, 16, 16]
        anomaly_maps = F.interpolate(
            distance.unsqueeze(1), size=images.shape[2:], mode="bilinear"
        ).squeeze(1)
        # threshold = 0.5
        threshold = self.pixel_threshold

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
            image_prediction = int(score >= self.image_threshold)
            self.image_predictions.append(image_prediction)

            mask_np = masks[sample_idx].squeeze().detach().cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            # self.pixel_scores_all.extend(amap.flatten().tolist())
            # self.pixel_labels_all.extend(mask_bin.flatten().tolist())
            pixel_prediction = (amap >= self.pixel_threshold).astype(np.uint8)

            self.pixel_scores_all.extend(amap.flatten().tolist())
            self.pixel_labels_all.extend(mask_bin.flatten().tolist())
            self.pixel_predictions_all.extend(pixel_prediction.flatten().tolist())

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
            # plt.imshow(amap, cmap="jet", vmin=0, vmax=1)
            plt.imshow(amap, cmap="jet")
            plt.title(f"Heatmap (Score: {score:.3f})")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(mask_np, cmap="gray")
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(amap > threshold, cmap="gray")
            # plt.title("Predicted")
            plt.title(f"Predicted\nThreshold={threshold:.4f}")
            plt.axis("off")

            plt.savefig(
                os.path.join(
                    self.output_path, f"{defect_type}_{batch_idx + sample_idx:03d}.png"
                )
            )
            plt.close()

    """
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
    """

    def on_predict_epoch_end(self) -> None:
        image_labels = np.asarray(self.image_labels)
        image_scores = np.asarray(self.image_scores)
        image_predictions = np.asarray(self.image_predictions)

        pixel_labels = np.asarray(self.pixel_labels_all)
        pixel_scores = np.asarray(self.pixel_scores_all)
        pixel_predictions = np.asarray(self.pixel_predictions_all)

        try:
            image_auroc = roc_auc_score(
                image_labels,
                image_scores,
            )
        except ValueError:
            image_auroc = float("nan")

        try:
            pixel_auroc = roc_auc_score(
                pixel_labels,
                pixel_scores,
            )
        except ValueError:
            pixel_auroc = float("nan")

        try:
            image_auprc = average_precision_score(
                image_labels,
                image_scores,
            )
        except ValueError:
            image_auprc = float("nan")

        try:
            pixel_auprc = average_precision_score(
                pixel_labels,
                pixel_scores,
            )
        except ValueError:
            pixel_auprc = float("nan")

        image_precision = precision_score(
            image_labels,
            image_predictions,
            zero_division=0,
        )
        image_recall = recall_score(
            image_labels,
            image_predictions,
            zero_division=0,
        )
        image_f1 = f1_score(
            image_labels,
            image_predictions,
            zero_division=0,
        )

        pixel_precision = precision_score(
            pixel_labels,
            pixel_predictions,
            zero_division=0,
        )
        pixel_recall = recall_score(
            pixel_labels,
            pixel_predictions,
            zero_division=0,
        )
        pixel_f1 = f1_score(
            pixel_labels,
            pixel_predictions,
            zero_division=0,
        )

        image_confusion = confusion_matrix(
            image_labels,
            image_predictions,
            labels=[0, 1],
        )

        pixel_confusion = confusion_matrix(
            pixel_labels,
            pixel_predictions,
            labels=[0, 1],
        )

        metrics = {
            "I-AUROC": image_auroc,
            "I-AUPRC": image_auprc,
            "I-Precision": image_precision,
            "I-Recall": image_recall,
            "I-F1": image_f1,
            "P-AUROC": pixel_auroc,
            "P-AUPRC": pixel_auprc,
            "P-Precision": pixel_precision,
            "P-Recall": pixel_recall,
            "P-F1": pixel_f1,
            "Pixel threshold": self.pixel_threshold,
            "Image threshold": self.image_threshold,
        }

        print("\n" + "=" * 55)
        print("Evaluation results")
        print("=" * 55)

        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")

        print("\nImage confusion matrix:")
        print(image_confusion)

        print("\nPixel confusion matrix:")
        print(pixel_confusion)

        metrics_path = os.path.join(
            self.output_path,
            "metrics_v1.txt",
        )

        with open(metrics_path, "w", encoding="utf-8") as file:
            for name, value in metrics.items():
                file.write(f"{name}: {value:.6f}\n")

            file.write("\nImage confusion matrix:\n")
            file.write(np.array2string(image_confusion))
            file.write("\n\nPixel confusion matrix:\n")
            file.write(np.array2string(pixel_confusion))
            file.write("\n")

        print(f"\nMetrics saved to: {metrics_path}")
            
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
    """
    Windows-safe DataLoader.

    num_workers=0 keeps loading in the main process and avoids spawning
    extra Python processes that reload PyTorch/CUDA DLLs.
    """
    num_workers = int(config["num_workers"])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=USE_CUDA,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_numpy_worker if num_workers > 0 else None,
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

@torch.no_grad()
def calibrate_thresholds(
        config: dict,
        lit_module: StudentLightning,
    ) -> tuple[float, float]:
        """
        使用正常 train images 校正 pixel-level 與 image-level threshold。

        不使用 test label，因此不會把測試答案洩漏給模型。
        """
        class_name = config["class_name"]

        print(
            f"\n[CALIBRATION] 使用 {class_name} 正常訓練資料校正 threshold..."
        )

        calibration_dataset = MVTecDataset(
            DATA_ROOT,
            class_name,
            phase="train",
            augment=None,
        )

        calibration_loader = make_loader(
            config,
            calibration_dataset,
            batch_size=1,
            shuffle=False,
        )

        device = lit_module.device
        lit_module.eval()

        pixel_scores: list[np.ndarray] = []
        image_scores: list[float] = []

        for batch in calibration_loader:
            clean_images = batch[0].to(device)

            distance = lit_module.feature_distance(clean_images)

            anomaly_maps = F.interpolate(
                distance.unsqueeze(1),
                size=clean_images.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            for anomaly_map in anomaly_maps:
                anomaly_map_np = (
                    anomaly_map.float().detach().cpu().numpy()
                )

                anomaly_map_np = gaussian_filter(
                    anomaly_map_np,
                    sigma=4,
                )

                flattened = anomaly_map_np.flatten()

                pixel_scores.append(flattened)

                top_k = min(
                    int(config["score_top_k"]),
                    flattened.size,
                )

                image_score = float(
                    np.partition(
                        flattened,
                        flattened.size - top_k,
                    )[-top_k:].mean()
                )

                image_scores.append(image_score)

        if not pixel_scores or not image_scores:
            raise RuntimeError("正常資料門檻校正失敗：沒有取得任何分數。")

        all_pixel_scores = np.concatenate(pixel_scores)

        pixel_threshold = float(
            np.quantile(
                all_pixel_scores,
                config["pixel_threshold_quantile"],
            )
        )

        image_threshold = float(
            np.quantile(
                np.asarray(image_scores),
                config["image_threshold_quantile"],
            )
        )

        lit_module.pixel_threshold = pixel_threshold
        lit_module.image_threshold = image_threshold

        print(f"[CALIBRATION] Pixel threshold: {pixel_threshold:.6f}")
        print(f"[CALIBRATION] Image threshold: {image_threshold:.6f}")

        return pixel_threshold, image_threshold

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
                "teacher_batch_size": 2,
                "student_batch_size": 2,
                "checkpoint_dir": "./checkpoints/calibrated_ts_smoke",
            }
        )

    L.seed_everything(config["seed"], workers=True)

    if args.smoke:
        output_path = f"./results/{config['class_name']}_calibrated_ts_smoke"
    else:
        output_path = f"./results/{config['class_name']}_calibrated_ts"
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
    # evaluate(config, student_module)
    calibrate_thresholds(config, student_module)
    evaluate(config, student_module)
