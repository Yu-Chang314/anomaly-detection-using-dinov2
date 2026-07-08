import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock


def _conv_stem(out_dim: int) -> nn.Sequential:
    """Small CNN mapping a 224x224 image to a 16x16 feature map (stride 14)."""
    return nn.Sequential(
        nn.Conv2d(3, out_dim // 4, kernel_size=7, stride=7),  # 224 -> 32
        nn.GELU(),
        nn.Conv2d(out_dim // 4, out_dim // 2, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(out_dim // 2, out_dim, kernel_size=3, stride=2, padding=1),  # 32 -> 16
        nn.GELU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.GELU(),
    )


class TeacherNet(nn.Module):
    """Compact CNN distilled to regress DINOv2 patch tokens.

    Output: [B, out_dim, 16, 16] for a 224x224 input, matching the
    16x16 token grid of DINOv2 ViT-L/14.
    """

    def __init__(self, out_dim: int = 1024, width: int = 384) -> None:
        super().__init__()
        self.stem = _conv_stem(width)
        self.proj = nn.Conv2d(width, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.stem(x))


class StudentNet(nn.Module):
    """Student with a CNN branch (local detail) and a Swin Transformer branch
    (global multi-scale context), fused to the teacher's feature dimension.
    """

    def __init__(
        self,
        out_dim: int = 1024,
        cnn_dim: int = 384,
        swin_dim: int = 384,
        swin_depth: int = 4,
        swin_heads: int = 6,
        window_size: int = 8,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.cnn = _conv_stem(cnn_dim)

        self.patch_embed = nn.Conv2d(3, swin_dim, kernel_size=14, stride=14)  # 224 -> 16
        self.swin_blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=swin_dim,
                    input_resolution=(grid_size, grid_size),
                    num_heads=swin_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
                for i in range(swin_depth)
            ]
        )
        self.swin_norm = nn.LayerNorm(swin_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(cnn_dim + swin_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)  # [B, cnn_dim, 16, 16]

        swin_feat = self.patch_embed(x).permute(0, 2, 3, 1)  # [B, 16, 16, swin_dim]
        for blk in self.swin_blocks:
            swin_feat = blk(swin_feat)
        swin_feat = self.swin_norm(swin_feat).permute(0, 3, 1, 2)  # [B, swin_dim, 16, 16]

        return self.fuse(torch.cat((cnn_feat, swin_feat), dim=1))
