import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pos_embed import get_2d_sincos_pos_embed
from block import TransformerBlock


class SegmentorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        n_cls: int,
        patch_size: int,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.scale = embed_dim**-0.5

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, embed_dim))
        self.proj_dec = nn.Linear(input_dim, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, dropout=dropout, drop_path=dpr[i]
                )
                for i in range(num_layers)
            ]
        )

        self.proj_patch = nn.Parameter(self.scale * torch.randn(embed_dim, embed_dim))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(embed_dim, embed_dim))

        self.decoder_norm = nn.RMSNorm(embed_dim)
        self.mask_norm = nn.RMSNorm(n_cls)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.RMSNorm):
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x: torch.Tensor, im_size: tuple[int, ...]) -> torch.Tensor:
        group_size = im_size[0] // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Split the output into patch features and class features
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        # L2 Normalization
        patches = F.normalize(patches, dim=-1, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=-1, p=2)

        # Scalar Product produces class map C in R^(n×2)
        masks = patches @ cls_seg_feat.transpose(1, 2)  # (B, N, n_cls)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=group_size)
        return masks


class ReconstructNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        num_patches: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, dim), requires_grad=False
        )  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.recon_blocks = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, dropout=dropout, drop_path=dpr[i])
                for i in range(num_layers)
            ]
        )

        self.recon_norm = nn.RMSNorm(dim)
        self.apply(self._init_weights)

        grid_size = int(math.sqrt(num_patches))
        assert grid_size**2 == num_patches, "num_patches should be a perfect square"
        pos_embed = get_2d_sincos_pos_embed(dim, grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.RMSNorm):
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {"pos_embed"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed

        for blk in self.recon_blocks:
            x = blk(x)
        x = self.recon_norm(x)
        return x
