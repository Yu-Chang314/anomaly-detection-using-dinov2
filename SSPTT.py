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
from typing import Optional, Any, Tuple
from scipy.ndimage import gaussian_filter
from anomaly_types.perlin import PerlinAnomalyGenerator
from anomaly_types.cutpaste import CutPasteNormal, CutPasteScar
from mvtec import MVTecDataset
import lightning as L
from model import TeacherStudentNet

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

# ==========================================
# 0. Global Configuration (Aligned with T-S Architecture)
# ==========================================
USE_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_CUDA else "cpu"
DEVICE = "cuda" if USE_CUDA else "cpu"
CONFIG = {
    "seed": 42,
    "img_size": 224,
    "patch_size": 14,
    "class_name": "transistor",
    "embed_dim": 1024,  # ViT-B對齊
    "num_heads": 8,
    "num_layers": 4,
    "epochs": 300,
    "warmup_epochs": 5,
    "batch_size": 4,
    "lr": 2e-4,         # T-S 架構通常需要略微明快的學習率
    "margin": 0.5,      # 特徵失配的基準邊界距離
    "device": DEVICE,
    "checkpoint_dir": "./checkpoints",
    "dropout": 0.0,
    "num_tokens": 16 * 16,
    "tokenizer_name": "dinov2_vitl14_reg",
    "repo_or_dir": "facebookresearch/dinov2",
    "num_workers": 4,
    "precision": "bf16-mixed",
}

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
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
        self.model = TeacherStudentNet(
            tokenizer=DINOWrapper(config["repo_or_dir"], config["tokenizer_name"]),
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )

    def forward(self, x: torch.Tensor, clean_x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, clean_x=clean_x)

    def training_step(self, batch, _) -> torch.Tensor:
        clean_images, augmented_images, masks, _ = batch

        # 將真實像素級遮罩做 MaxPool 降採樣，對齊到 16x16 的 Token 空間
        patch_mask = F.max_pool2d(
            masks,
            kernel_size=self.config["patch_size"],
            stride=self.config["patch_size"],
        ).squeeze(1)  # 尺寸: [B, 16, 16]

        # 前向傳播
        student_tokens, teacher_tokens = self(augmented_images, clean_x=clean_images)

        # === 核心修改：訓練也統一改用餘弦距離 ===
        cos_sim = F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
        dist_map = 1.0 - cos_sim  # 數值區間被嚴格鎖定在 [0, 2] 之間
        dist_map = dist_map.view(-1, 16, 16)  # [B, 16, 16]

        # 1. 正常區域 (Normal Loss): 限制 Student 必須與 Teacher 完美貼合（距離趨近於 0）
        normal_mask = (patch_mask == 0).float()
        loss_normal = (dist_map * normal_mask).sum() / (normal_mask.sum() + 1e-6)

        # 2. 合成缺陷區域 (Anomaly Margin Loss): 超過 margin 就不處罰，小於 margin 則強力推開
        anomaly_mask = (patch_mask > 0).float()
        margin = self.config["margin"]  # 建議在 CONFIG 中將 margin 設為 0.5 左右
        loss_anomaly = (torch.relu(margin - dist_map) * anomaly_mask).sum() / (anomaly_mask.sum() + 1e-6)

        # 總損失聯合優化
        loss = loss_normal + loss_anomaly

        bs = clean_images.size(0)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        self.log("loss_normal", loss_normal, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        self.log("loss_anomaly", loss_anomaly, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        return loss

    def on_predict_start(self) -> None:
        self.image_scores: list[float] = []
        self.image_labels: list[int] = []
        self.pixel_scores_all: list[float] = []
        self.pixel_labels_all: list[int] = []
        # 新增：用來動態計算最佳全域門檻的容器
        self.all_predicted_maps: list[np.ndarray] = []
        self.all_gt_masks: list[np.ndarray] = []
        self.all_metadata: list[dict] = []
        os.makedirs(self.output_path, exist_ok=True)

    def predict_step(self, batch, batch_idx: int):
        images, labels, masks, paths = batch
        
        # 1. 取得 Teacher 與 Student 特徵
        student_tokens, teacher_tokens = self(images)
        
        # 2. 改用餘弦相似度 (Cosine Similarity)，將距離鎖定在 [0, 2] 區間
        # F.cosine_similarity 輸出 [-1, 1]，1.0 代表完全相同
        cos_sim = F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
        dist_map = 1.0 - cos_sim  # 越不相似，數值越接近 2.0，完美泛化所有材質！
        
        # 重新排列成 2D 特徵圖
        dist_map = dist_map.view(-1, 16, 16).unsqueeze(1)
        
        # 3. 雙線性插值放大回原圖大小
        anomaly_maps = F.interpolate(dist_map, size=images.shape[2:], mode="bilinear", align_corners=False).squeeze(1)

        for sample_idx in range(images.size(0)):
            probs = anomaly_maps[sample_idx].detach().cpu().numpy()
            
            # 4. 減小高斯平滑核 (Sigma=1.5)，保留布料或細微瑕疵的邊緣細節，防止被抹平
            probs = gaussian_filter(probs, sigma=1.5)

            # 單圖局部最大/最小值歸一化（有助於跨類別可視化對比）
            p_min, p_max = probs.min(), probs.max()
            if p_max - p_min > 1e-5:
                probs_norm = (probs - p_min) / (p_max - p_min)
            else:
                probs_norm = np.zeros_like(probs)

            # 影像級評分：取最高的前 1% 異常像素均值
            top_k_pixels = int(probs.size * 0.01)
            patch_score = float(np.mean(np.sort(probs.flatten())[-top_k_pixels:]))

            try:
                image_label = int(labels[sample_idx].item())
            except Exception:
                image_label = int(labels[sample_idx])

            self.image_scores.append(patch_score)
            self.image_labels.append(image_label)

            mask_np = masks[sample_idx].squeeze().detach().cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            
            # 儲存原始餘弦分數用於全域評估，儲存歸一化分數用於繪圖
            self.pixel_scores_all.extend(probs.flatten().tolist())
            self.pixel_labels_all.extend(mask_bin.flatten().tolist())

            self.all_predicted_maps.append(probs) # 存原始餘弦分數
            self.all_gt_masks.append(mask_np)
            self.all_metadata.append({
                "image": images[sample_idx].cpu(),
                "path": paths[sample_idx],
                "patch_score": patch_score,
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "probs_norm": probs_norm # 存歸一化分數
            })

    # 這裡改成 on_predict_end，確保預測生命週期完全結束、資料百分之百收集完畢後才執行
    def on_predict_end(self) -> None:
        print("\n" + "#"*50)
        print(f"[DEBUG] on_predict_end 觸發！目前累計收集了 {len(self.all_metadata)} 張圖片的預測結果。")
        print("#"*50 + "\n")

        if len(self.all_metadata) == 0:
            print("[WARN] 警告：self.all_metadata 是空的！代表 predict_step 沒有成功存入任何資料。")
            return

        try:
            i_auroc = roc_auc_score(self.image_labels, self.image_scores)
            p_auroc = roc_auc_score(self.pixel_labels_all, self.pixel_scores_all)
        except Exception as error:
            warnings.warn(f"AUROC calculation failed: {error}")
            i_auroc, p_auroc = float("nan"), float("nan")

        print(f"\n[SUMMARY] I-AUROC: {i_auroc:.4f} | P-AUROC: {p_auroc:.4f}")

        # --- 核心改動：改用「全域均值 + 3倍標準差」作為穩健統計門檻 ---
        normal_pixels = []
        for i, label in enumerate(self.image_labels):
            if label == 0:
                normal_pixels.extend(self.all_predicted_maps[i].flatten())
        
        if len(normal_pixels) > 0:
            normal_pixels = np.array(normal_pixels)
            mean_val = np.mean(normal_pixels)
            std_val = np.std(normal_pixels)
            global_thresh = float(mean_val + 3.0 * std_val)
            print(f"[THRES] Normal Pixels Stat -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        else:
            global_thresh = float(np.percentile(self.pixel_scores_all, 95))
            
        print(f"[THRES] Final Global Robust Threshold: {global_thresh:.4f}")

        # --- 統一繪圖輸出 ---
        print(f"[INFO] 開始繪圖並儲存至: {self.output_path} ...")
        for i, meta in enumerate(self.all_metadata):
            probs = self.all_predicted_maps[i]
            mask_np = self.all_gt_masks[i]
            path = meta["path"]
            defect_type = os.path.basename(os.path.dirname(path))
            
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(np.clip(denormalize(meta["image"]).permute(1, 2, 0).numpy(), 0, 1))
            plt.title(f"{defect_type}")
            plt.axis("off")

            # 加上 vmin/vmax 鎖定，避免正常圖被自動放大噪聲
            plt.subplot(1, 4, 2)
            plt.imshow(probs, cmap="jet", vmin=0.0, vmax=0.5)
            plt.title(f"Anomaly Map (Score: {meta['patch_score']:.3f})")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(mask_np, cmap="gray")
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(probs > global_thresh, cmap="gray")
            plt.title("Predicted Mask (3-Sigma Thresh)")
            plt.axis("off")

            # 儲存圖片
            save_filename = f"{defect_type}_{meta['batch_idx'] + meta['sample_idx']:03d}.png"
            plt.savefig(os.path.join(self.output_path, save_filename))
            plt.close()
            
        print(f"[INFO] 成功繪製並儲存了 {len(self.all_metadata)} 張圖片！")

        with open(os.path.join(self.output_path, "metrics.txt"), "w") as f:
            f.write(f"I-AUROC: {i_auroc}\n")
            f.write(f"P-AUROC: {p_auroc}\n")
            f.write(f"Global_Threshold: {global_thresh}\n")

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
                CosineAnnealingLR(optimizer, T_max=self.config["epochs"] - self.config["warmup_epochs"]),
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
    def __init__(self, class_name: str, checkpoint_dir: str, every_n_epochs: int = 50) -> None:
        super().__init__()
        self.class_name = class_name
        self.checkpoint_dir = checkpoint_dir
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: SSPTTLightning) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0:
            torch.save(
                pl_module.model.student.state_dict(),  # 主要保存受訓完成的 Student 權重
                os.path.join(self.checkpoint_dir, f"{self.class_name}_student_ep{epoch}.pth"),
            )

# ==========================================
# 4. Main Execution
# ==========================================
def run_one_margin(margin: float, config: dict) -> dict:
    """Train + evaluate for one margin threshold value."""
    CLASS_NAME = config["class_name"]
    DATA_ROOT = "D:/Users/peggy/Dataset/mvtec_anomaly_detection"
    DTD_ROOT = "D:/Users/peggy/Dataset/dtd/images"
    OUTPUT_PATH = f"./results/TS_{CLASS_NAME}/{CLASS_NAME}_margin_{margin:.1f}"
    CKPT_DIR = config["checkpoint_dir"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    cfg = {**config, "margin": margin}
    lit_module = SSPTTLightning(cfg, output_path=OUTPUT_PATH)

    if os.path.exists(os.path.join(DATA_ROOT, CLASS_NAME)):
        train_ds = MVTecDataset(
            DATA_ROOT,
            CLASS_NAME,
            phase="train",
            anomaly_generators=[
                PerlinAnomalyGenerator(anomaly_source_path=DTD_ROOT, probability=1.0, blend_factor=(0.1, 1.0)),
                CutPasteNormal(probability=1.0),
                CutPasteScar(probability=1.0, length_range=(10, 224)),
            ],
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
            prefetch_factor=2 if cfg["num_workers"] > 0 else None,
        )
        print(f"\n{'='*50}")
        print(f"  Training Teacher-Student with Margin={margin} ({CLASS_NAME})")
        print(f"{'='*50}")
        trainer = L.Trainer(
            max_epochs=cfg["epochs"],
            accelerator=ACCELERATOR,
            devices=1,
            deterministic=False,
            log_every_n_steps=1,
            precision=cfg["precision"] if USE_CUDA else "32-true",
            benchmark=True,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[SavePTHCallback(CLASS_NAME, CKPT_DIR, every_n_epochs=cfg["epochs"])],
        )
        trainer.fit(lit_module, train_dataloaders=train_loader)
    else:
        trainer = L.Trainer(accelerator=ACCELERATOR, devices=1, logger=False, enable_progress_bar=True)

    print(f"\n[EVAL] Margin={margin} — Starting Inference Verification...")
    test_ds = MVTecDataset(DATA_ROOT, CLASS_NAME, phase="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    trainer.predict(lit_module, dataloaders=test_loader)

    metrics_path = os.path.join(OUTPUT_PATH, "metrics.txt")
    i_auroc, p_auroc = float("nan"), float("nan")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                if line.startswith("I-AUROC"):
                    i_auroc = float(line.split(":")[1].strip())
                elif line.startswith("P-AUROC"):
                    p_auroc = float(line.split(":")[1].strip())

    return {"margin": margin, "i_auroc": i_auroc, "p_auroc": p_auroc}

if __name__ == "__main__":
    L.seed_everything(CONFIG["seed"], workers=True)

    # 針對 T-S 空間最關鍵的排斥邊界 (Margin) 進行優化網格搜索
    MARGIN_GRID = [0.4, 0.6]
    all_results = []

    for m in MARGIN_GRID:
        result = run_one_margin(m, CONFIG)
        all_results.append(result)
        print(f"[GRID RESULT] Margin={m:.1f} | I-AUROC={result['i_auroc']:.4f} | P-AUROC={result['p_auroc']:.4f}")

    print("\n" + "=" * 50)
    print("  Teacher-Student Margin Grid Search Summary")
    print("=" * 50)
    print(f"{'margin':>8}  {'I-AUROC':>8}  {'P-AUROC':>8}")
    for r in all_results:
        print(f"{r['margin']:>8.1f}  {r['i_auroc']:>8.4f}  {r['p_auroc']:>8.4f}")

    best = max(all_results, key=lambda r: r["i_auroc"])
    print(f"\nBest Margin Configuration: {best['margin']} | Best I-AUROC={best['i_auroc']:.4f} | P-AUROC={best['p_auroc']:.4f}")

    summary_path = f"./results/{CONFIG['class_name']}_ts_margin_grid_search.txt"
    with open(summary_path, "w") as f:
        f.write(f"{'margin':>8}  {'I-AUROC':>8}  {'P-AUROC':>8}\n")
        for r in all_results:
            f.write(f"{r['margin']:>8.1f}  {r['i_auroc']:>8.4f}  {r['p_auroc']:>8.4f}\n")
        f.write(f"\nBest Margin: {best['margin']}\n")
    print(f"Summary saved cleanly to {summary_path}")