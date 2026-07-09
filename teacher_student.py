import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import PatchMerging, SwinTransformerBlock


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
    16x16 token grid of DINOv2 ViT-L/14. Kept deliberately small so
    inference stays fast.
    """

    def __init__(self, out_dim: int = 1024, width: int = 384) -> None:
        super().__init__()
        self.stem = _conv_stem(width)
        self.proj = nn.Conv2d(width, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.stem(x))


class DualUpsample(nn.Module):
    """MSTUnet's upsample block: bilinear interpolation and PixelShuffle paths,
    each producing [B, dim/2, 2H, 2W], summed. (Jiang et al., TII 2023, Sec. III-B)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bilinear_proj = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.shuffle_proj = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_bilinear = self.bilinear_proj(
            F.interpolate(x, scale_factor=2, mode="bilinear")
        )
        up_shuffle = self.shuffle(self.shuffle_proj(x))
        return up_bilinear + up_shuffle


class SwinStage(nn.Module):
    """A stack of Swin blocks at one resolution, alternating W-MSA / SW-MSA."""

    def __init__(
        self, dim: int, resolution: int, depth: int, num_heads: int, window_size: int
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=(resolution, resolution),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # NHWC
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinUNetBranch(nn.Module):
    """Multi-scale U-shaped Swin branch modeled on MSTUnet's inpainting network:
    Swin stages with patch merging on the way down, dual upsampling with a skip
    connection on the way up. The patch embed uses stride 7 so the output grid
    (16x16) aligns with the teacher's DINOv2 token grid.

        224 --k7s7--> 32x32(dim) --merge--> 16x16(2*dim) --merge--> 8x8(4*dim)
                                     |                                 |
                                    skip <---------- dual upsample ----+
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depths: tuple[int, int, int] = (2, 2, 2),
        num_heads: tuple[int, int, int] = (6, 12, 24),
        window_size: int = 8,
    ) -> None:
        super().__init__()
        dims = (embed_dim, embed_dim * 2, embed_dim * 4)
        self.out_dim = dims[1]

        self.patch_embed = nn.Conv2d(3, dims[0], kernel_size=7, stride=7)  # 224 -> 32

        self.stage1 = SwinStage(dims[0], 32, depths[0], num_heads[0], window_size)
        self.merge1 = PatchMerging(dims[0])  # 32 -> 16, dims[1]
        self.stage2 = SwinStage(dims[1], 16, depths[1], num_heads[1], window_size)
        self.merge2 = PatchMerging(dims[1])  # 16 -> 8, dims[2]
        self.stage3 = SwinStage(dims[2], 8, depths[2], num_heads[2], window_size)

        self.upsample = DualUpsample(dims[2])  # 8 -> 16, dims[1]
        self.skip_fuse = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1)
        self.norm = nn.LayerNorm(dims[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # [B, 32, 32, dim]
        x = self.merge1(self.stage1(x))  # [B, 16, 16, 2*dim]
        skip = self.stage2(x)  # [B, 16, 16, 2*dim]
        x = self.merge2(skip)  # [B, 8, 8, 4*dim]
        x = self.stage3(x).permute(0, 3, 1, 2)  # [B, 4*dim, 8, 8]

        x = self.upsample(x)  # [B, 2*dim, 16, 16]
        x = self.skip_fuse(torch.cat((x, skip.permute(0, 3, 1, 2)), dim=1))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x  # [B, 2*dim, 16, 16]


class StudentNet(nn.Module):
    """Student with a CNN branch (local detail) and a multi-scale Swin U-Net
    branch (global context, MSTUnet-style), fused to the teacher's feature
    dimension.
    """

    def __init__(
        self,
        out_dim: int = 1024,
        cnn_dim: int = 384,
        swin_embed_dim: int = 192,
        swin_depths: tuple[int, int, int] = (2, 2, 2),
        swin_heads: tuple[int, int, int] = (6, 12, 24),
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.cnn = _conv_stem(cnn_dim)
        self.swin = SwinUNetBranch(swin_embed_dim, swin_depths, swin_heads, window_size)

        self.fuse = nn.Sequential(
            nn.Conv2d(cnn_dim + self.swin.out_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)  # [B, cnn_dim, 16, 16]
        swin_feat = self.swin(x)  # [B, 2*swin_embed_dim, 16, 16]
        return self.fuse(torch.cat((cnn_feat, swin_feat), dim=1))
