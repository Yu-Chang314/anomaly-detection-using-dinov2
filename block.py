import torch
import torch.nn as nn
from timm.layers import DropPath
from nystrom import NystromAttention
import torch.nn.functional as F
from typing import Optional, Callable


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        swiglu_hidden_features = int(2 * hidden_features / 3)
        swiglu_hidden_features = (swiglu_hidden_features + 7) // 8 * 8

        self.w12 = nn.Linear(in_features, swiglu_hidden_features * 2, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_landmarks: int = 64,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = NystromAttention(
            dropout,
            dim,
            num_heads,
            num_landmarks,
        )
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn_layer = SwiGLUFFN(
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim,
            bias=True,
        )
        # self.ffn_layer = Mlp(dim, dim * 4, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.attn_norm(x)))
        x = x + self.drop_path(self.ffn_layer(self.ffn_norm(x)))
        return x
