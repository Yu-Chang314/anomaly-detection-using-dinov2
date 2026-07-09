import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, Tuple

class StudentTransformerBlock(nn.Module):
    """
    自包含的輕量化 Transformer 區塊，免去外部檔案依賴。
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformerStudent(nn.Module):
    """
    具備區域感知的 Student 網路：
    結合 ResNet 前期特徵（強化局部幾何紋理）與 Transformer 區塊（對齊 Teacher 的全局表徵）。
    """
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        # 使用標準 ResNet18 提取豐富的低階與中階局部特徵
        res = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn_stem = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1,  # 空間解析度 /4
            res.layer2,  # 空間解析度 /8
            res.layer3   # 空間解析度 /16 (輸入 224x224 時輸出為 14x14)
        )
        
        # 強制將 CNN 空間尺寸自適應調整為 16x16，完美對齊 DINOv2 的 Token 數量 (16*16=256)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.proj = nn.Linear(256, embed_dim)  # ResNet18 layer3 的 Channel 是 256
        
        # 可學習的位置編碼
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 輕量化 Transformer 堆疊
        self.blocks = nn.ModuleList([
            StudentTransformerBlock(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_stem(x)              # 輸出尺寸: [B, 256, 14, 14]
        x = self.pool(x)                  # 修正尺寸: [B, 256, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # 展平排列: [B, 256, 256]
        x = self.proj(x)                  # 維度映射: [B, 256, 1024]
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class TeacherStudentNet(nn.Module):
    """
    完整的不一致性偵測雙軌網路（Teacher-Student Network）
    """
    def __init__(self, tokenizer: nn.Module, embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.teacher = tokenizer
        # 鐵面考官：嚴格凍結 Teacher 權重，不參與梯度更新
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = CNNTransformerStudent(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, clean_x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            # 訓練階段：Teacher 唯讀最純淨的原始圖；Student 觀察被擾動污染的增強圖
            with torch.no_grad():
                teacher_feats = self.teacher(clean_x if clean_x is not None else x)
            student_feats = self.student(x)
            return student_feats, teacher_feats
        else:
            # 推論階段：雙軌同圖輸入，兩者吃完全同一張待測影像
            with torch.no_grad():
                teacher_feats = self.teacher(x)
            student_feats = self.student(x)
            return student_feats, teacher_feats