import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import PatchMerging, SwinTransformerBlock
from timm.models.vision_transformer import Block as ViTBlock

from pos_embed import get_2d_sincos_pos_embed


class ResidualBlock(nn.Module):
    """Pre-activation 3x3-3x3 residual conv block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class ConvEncoder(nn.Module):
    """CNN mapping a 224x224 image to a 16x16 feature map (stride 14),
    with `depth` residual blocks at each of the 32x32 and 16x16 scales."""

    def __init__(self, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        half = out_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, half, kernel_size=7, stride=7),  # 224 -> 32
            nn.GELU(),
            nn.Conv2d(half, half, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(*[ResidualBlock(half) for _ in range(depth)])
        self.down = nn.Conv2d(half, out_dim, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.stage2 = nn.Sequential(*[ResidualBlock(out_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.down(self.stage1(self.stem(x))))


class TeacherNet(nn.Module):
    """Compact CNN + global-attention hybrid distilled to regress DINOv2
    patch tokens.

    DINOv2 patch tokens carry global context (every token attends to the
    whole image), so a local-receptive-field CNN alone has an irreducible
    regression error; a few ViT blocks over the 16x16 grid supply that
    global mixing while keeping inference fast (256 tokens only).

    Output: [B, out_dim, 16, 16] for a 224x224 input, matching the
    16x16 token grid of DINOv2 ViT-L/14.
    """

    def __init__(
        self,
        out_dim: int = 1024,
        width: int = 512,
        conv_depth: int = 2,
        attn_depth: int = 4,
        num_heads: int = 8,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(width, depth=conv_depth)
        pos = get_2d_sincos_pos_embed(width, grid_size)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos).float().unsqueeze(0)
        )
        self.blocks = nn.ModuleList(
            [ViTBlock(dim=width, num_heads=num_heads) for _ in range(attn_depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)  # [B, width, 16, 16]
        b, _, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.proj(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, -1, h, w)


class PatchPartition(nn.Module):
    """MSTUnet's patch partition (Eq. 4-5): a 3x3 conv extracts shallow
    features, then a strided conv partitions into patches. Stride 7 (instead
    of the paper's 4) so the U-shape's levels align with the teacher's
    16x16 DINOv2 token grid: 224 -> 32 -> 16 -> 8 -> 4.
    """

    def __init__(self, in_ch: int = 3, dim: int = 96) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=7, stride=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> NHWC
        return self.proj(self.conv(x)).permute(0, 2, 3, 1)


class UpsampleBlock(nn.Module):
    """MSTUnet's upsample block (Sec. III-B): bilinear interpolation and
    PixelShuffle paths, each producing [2H, 2W, out_ch], summed.
    """

    def __init__(self, in_ch: int, out_ch: int | None = None) -> None:
        super().__init__()
        out_ch = in_ch // 2 if out_ch is None else out_ch
        self.conv_bilinear = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv_pixel = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # NHWC -> NHWC
        x = x.permute(0, 3, 1, 2)
        u_prime = F.interpolate(
            self.conv_bilinear(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        u_double_prime = self.pixel_shuffle(self.conv_pixel(x))
        return (u_prime + u_double_prime).permute(0, 2, 3, 1)


class SwinStage(nn.Module):
    """A stack of Swin blocks at one resolution, alternating W-MSA / SW-MSA.

    All feature grids (32/16/8/4) are divisible by (or smaller than) the
    window, so no boundary padding is needed; timm shrinks the window when
    the grid is smaller than window_size.
    """

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
    """Multi-scale U-shaped Swin branch following MSTUnet's inpainting network
    (Jiang et al., TII 2023): a four-level Swin encoder with patch merging,
    and a Swin decoder with dual upsampling and skip connections. The decoder
    stops at 16x16 to stay aligned with the teacher's DINOv2 token grid
    instead of reconstructing pixels.

        224 -PatchPartition-> 32(C) -> 16(2C) -> 8(4C) -> 4(8C)
                               |        |         |        |
                              skip     skip      dec1 <----+
                               x        +-------- dec2
                            (unused)         out: 16x16x2C
    """

    def __init__(
        self,
        embed_dim: int = 96,
        depth: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        c = embed_dim
        heads = (c // 32, c // 16, c // 8, c // 4)  # 3/6/12/24 for C=96
        self.out_dim = 2 * c

        self.patch_partition = PatchPartition(3, c)
        self.enc1 = SwinStage(c, 32, depth, heads[0], window_size)
        self.down1 = PatchMerging(c)
        self.enc2 = SwinStage(2 * c, 16, depth, heads[1], window_size)
        self.down2 = PatchMerging(2 * c)
        self.enc3 = SwinStage(4 * c, 8, depth, heads[2], window_size)
        self.down3 = PatchMerging(4 * c)
        self.enc4 = SwinStage(8 * c, 4, depth, heads[3], window_size)

        self.up1 = UpsampleBlock(8 * c, 4 * c)
        self.concat_proj1 = nn.Linear(8 * c, 4 * c)
        self.dec1 = SwinStage(4 * c, 8, depth, heads[2], window_size)

        self.up2 = UpsampleBlock(4 * c, 2 * c)
        self.concat_proj2 = nn.Linear(4 * c, 2 * c)
        self.dec2 = SwinStage(2 * c, 16, depth, heads[1], window_size)

        self.norm = nn.LayerNorm(2 * c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.patch_partition(x))  # [B, 32, 32, C]
        e2 = self.enc2(self.down1(e1))  # [B, 16, 16, 2C]
        e3 = self.enc3(self.down2(e2))  # [B, 8, 8, 4C]
        e4 = self.enc4(self.down3(e3))  # [B, 4, 4, 8C]

        d1 = self.dec1(self.concat_proj1(torch.cat([self.up1(e4), e3], dim=-1)))
        d2 = self.dec2(self.concat_proj2(torch.cat([self.up2(d1), e2], dim=-1)))

        return self.norm(d2).permute(0, 3, 1, 2)  # [B, 2C, 16, 16]


class StudentNet(nn.Module):
    """Student with a CNN branch (local detail) and a multi-scale Swin U-Net
    branch (global context, MSTUnet-style), fused to the teacher's feature
    dimension.
    """

    def __init__(
        self,
        out_dim: int = 1024,
        cnn_dim: int = 512,
        cnn_depth: int = 2,
        swin_embed_dim: int = 128,
        swin_depth: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.cnn = ConvEncoder(cnn_dim, depth=cnn_depth)
        self.swin = SwinUNetBranch(swin_embed_dim, swin_depth, window_size)

        self.fuse = nn.Sequential(
            nn.Conv2d(cnn_dim + self.swin.out_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)  # [B, cnn_dim, 16, 16]
        swin_feat = self.swin(x)  # [B, 2*swin_embed_dim, 16, 16]
        return self.fuse(torch.cat((cnn_feat, swin_feat), dim=1))
