import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from typing import Optional


def scaled_query_key_softmax(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    return F.softmax(attn_weight, dim=-1)


def iterative_pinv(softmax_mat: torch.Tensor, n_iter=6, pinverse_original_init=False):
    """
    Computing the Moore-Penrose inverse.
    Use an iterative method from (Razavi et al. 2014) to approximate the Moore-Penrose inverse via efficient
    matrix-matrix multiplications.

    Notice: This function assumes that the matrix has had softmax applied to it.
    """

    i = torch.eye(
        softmax_mat.size(-1), device=softmax_mat.device, dtype=softmax_mat.dtype
    )
    k = softmax_mat

    # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
    if pinverse_original_init:
        # This original implementation is more conservative to compute coefficient of Z_0.
        v = 1 / torch.max(torch.sum(k, dim=-2)) * k.transpose(-1, -2)
    else:
        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster
        # convergence.
        v = (
            1
            / torch.max(torch.sum(k, dim=-2), dim=-1).values[:, None, None]
            * k.transpose(-1, -2)
        )

    for _ in range(n_iter):
        kv = torch.matmul(k, v)
        v = torch.matmul(
            0.25 * v,
            13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)),
        )
    return v


class AvgPool(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor):
        # Average independently for every segment in the sequence dimension
        seq_len = x.shape[1]
        head_dim = x.shape[2]
        segments = seq_len // self.n
        assert segments > 0, "num_landmarks should be smaller than the sequence length"

        # Dimensions are a match
        if seq_len % self.n == 0:
            return x.reshape(
                -1,
                self.n,
                segments,
                head_dim,
            ).mean(dim=-2)

        # Handle the last segment boundary being off
        n_round = self.n - seq_len % self.n

        x_avg_round = (
            x[:, : n_round * segments, :]
            .reshape(-1, n_round, segments, head_dim)
            .mean(dim=-2)
        )
        x_avg_off = (
            x[:, n_round * segments :, :]
            .reshape(-1, self.n - n_round, segments + 1, head_dim)
            .mean(dim=-2)
        )
        return torch.cat((x_avg_round, x_avg_off), dim=-2)


class AvgPool1d(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = F.adaptive_avg_pool1d(x.transpose(1, 2), self.n)
        x_avg = x_avg.transpose(1, 2)
        return x_avg


class NystromAttention(nn.Module):
    """Nyström attention mechanism from Nystromformer_.
    ::
        Xiong et al., "A Nyström-based Algorithm for Approximating Self-Attention", 2021.

    .. _Nystromformer: https://arxiv.org/pdf/2102.03902.pdf

    This code is based on Xformers_ but simplified here.
    .. _Xformers: https://github.com/AbdBarho/xformers-wheels/blob/main/xformers/components/attention/nystrom.py
    """

    def __init__(
        self,
        dropout: float,
        dim: int,
        num_heads: int,
        num_landmarks: int = 64,
        landmark_pooling: Optional[nn.Module] = None,
        bias: bool = True,
        use_razavi_pinverse: bool = True,
        pinverse_original_init: bool = False,
        inv_iterations: int = 6,  # recommended default in paper was 6.
        v_skip_connection: Optional[nn.Module] = None,
        conv_kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.inv_iterations = inv_iterations
        self.use_razavi_pinverse = use_razavi_pinverse
        self.pinverse_original_init = pinverse_original_init
        self.skip_connection = v_skip_connection

        head_dim = dim // num_heads
        assert head_dim > 0, "num_heads should be less than or equal to dim"

        inner_dim = head_dim * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.attn_proj = nn.Linear(inner_dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

        if self.skip_connection is None and conv_kernel_size is not None:
            self.skip_connection = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

        if landmark_pooling is None:
            self.landmark_pooling = AvgPool(self.num_landmarks)
        else:
            self.landmark_pooling = landmark_pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get query, key, value and reshape for multi-head attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b s (h d) -> (b h) s d", h=self.num_heads),
            qkv,
        )

        seq_len = k.size(-2)
        if self.num_landmarks >= seq_len:
            x_out = F.scaled_dot_product_attention(q, k, v)
        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)

            kernel_1 = scaled_query_key_softmax(q, k_landmarks)
            kernel_2 = scaled_query_key_softmax(q_landmarks, k_landmarks)
            kernel_3 = F.scaled_dot_product_attention(q_landmarks, k, v)

            kernel_2_inv = (
                iterative_pinv(
                    kernel_2,
                    n_iter=self.inv_iterations,
                    pinverse_original_init=self.pinverse_original_init,
                )
                if self.use_razavi_pinverse
                else torch.linalg.pinv(kernel_2)
            )

            x_out = torch.matmul(torch.matmul(kernel_1, kernel_2_inv), kernel_3)

        if self.skip_connection:
            # Assumption here is that v is 3D (N, S, D).
            # Reshape to (Batch, Heads, S, D) for Conv2d
            v_conv = self.skip_connection(
                v.reshape(-1, self.num_heads, v.size(-2), v.size(-1))
            )
            # Add back to x_out (flatten Batch and Heads back to N)
            x_out += v_conv.reshape(-1, v_conv.size(-2), v_conv.size(-1))

        x_out = self.attn_drop(x_out)
        x_out = rearrange(x_out, "(b h) n d -> b n (h d)", h=self.num_heads)
        x_out = self.attn_proj(x_out)
        return x_out
