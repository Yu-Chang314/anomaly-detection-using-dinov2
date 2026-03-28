import torch
from torch import nn
import torch.nn.functional as F
from subnetworks import ReconstructNetwork, SegmentorNetwork
from typing import Optional


class SSPTT(nn.Module):
    def __init__(
        self,
        tokenizer: nn.Module,
        dim: int,
        patch_size: int,
        num_patches: int,
        mask_ratio: float,
        num_heads: int,
        num_layers: int,
        dropout: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        self.recon_net = ReconstructNetwork(
            dim,
            num_patches,
            num_heads,
            num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        self.segm_net = SegmentorNetwork(
            input_dim=dim * 2,
            embed_dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            n_cls=2,  # normal, abnormal
            patch_size=patch_size,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        self.mask_token = nn.Parameter(torch.randn(dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mask_token"}

    def random_masking(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape  # batch, sequence length, embedding dim
        num_masked = int(self.num_patches * self.mask_ratio)

        # Generate random indices per batch
        rand_indices = torch.randn(B, N, device=x.device).argsort(dim=1)
        masked_indices = rand_indices[:, :num_masked]  # [B, num_masked]

        mask_tokens = self.mask_token[None, None, :].repeat(
            B, num_masked, 1
        )  # [B, num_masked, dim]
        batch_indices = torch.arange(B, device=x.device).unsqueeze(-1)  # [B, 1]

        x[batch_indices, masked_indices, :] = mask_tokens
        return x

    def forward(
        self,
        x: torch.Tensor,
        clean_x: Optional[torch.Tensor] = None,
        return_patch_level_masks: bool = True,
    ) -> torch.Tensor:

        with torch.no_grad():
            x_tokens = self.tokenizer(x)
            recon_tokens = x_tokens.clone().detach()
            if self.training:
                clean_tokens = self.tokenizer(clean_x) if clean_x is not None else None
                recon_tokens = self.random_masking(recon_tokens)

        recon_tokens = self.recon_net(recon_tokens)
        segm_cat = torch.cat((recon_tokens, x_tokens), dim=-1)  # [B, N, 2*D]
        masks = self.segm_net(segm_cat, im_size=x.shape[2:])

        if not return_patch_level_masks:
            masks = F.interpolate(masks, size=x.shape[2:], mode="bilinear")

        if self.training:
            return masks, recon_tokens, clean_tokens
        else:
            return masks
