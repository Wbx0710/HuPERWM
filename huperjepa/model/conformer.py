"""Causal Conformer building blocks.

Provides the two primitives used by ComparisonRefinementEncoder:
- ConformerBlock  — a single Conformer layer (optionally causal).
- CausalConformerEncoder — a stack of causal ConformerBlocks with learned
  positional embeddings, used to produce the streaming language prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvModule(nn.Module):
    """Depthwise-separable convolution module from Conformer.

    When *causal=True* the depthwise conv uses left-only padding so that
    position k only sees positions 0..k, matching the causal attention mask.

    Uses GroupNorm (groups=32 or dim if dim<32) for non-causal and LayerNorm
    for causal — both are independent of padding positions and batch size.
    """

    def __init__(
        self, dim: int, kernel_size: int = 15, dropout: float = 0.1, causal: bool = False
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.causal = causal
        if causal:
            self.dw = nn.Conv1d(dim, dim, kernel_size, padding=0, groups=dim)
            self._causal_pad = kernel_size - 1
        else:
            self.dw = nn.Conv1d(
                dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
            )
        if causal:
            self.conv_norm = nn.LayerNorm(dim)
        else:
            num_groups = min(32, dim)
            while dim % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.conv_norm = nn.GroupNorm(num_groups, dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, H, K)
        x = self.pw1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)  # GLU
        if self.causal:
            x = F.pad(x, (self._causal_pad, 0))
        x = self.dw(x)
        if self.causal:
            x = x.transpose(1, 2)
            x = self.conv_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.conv_norm(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x.transpose(1, 2)  # (B, K, H)


class ConformerBlock(nn.Module):
    """Full Conformer block: FFN/2 → MHSA → Conv → FFN/2 → LN."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ff_dim: int,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.ffn1 = _FeedForward(dim, ff_dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.conv = _ConvModule(dim, conv_kernel, dropout, causal=causal)
        self.ffn2 = _FeedForward(dim, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.causal = causal

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)

        h = self.attn_norm(x)
        attn_mask = None
        if self.causal:
            K = x.shape[1]
            attn_mask = torch.triu(
                torch.full((K, K), float("-inf"), device=x.device), diagonal=1
            )
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + h

        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class CausalConformerEncoder(nn.Module):
    """Stack of causal ConformerBlocks for the streaming language prior.

    Each slot k only attends to slots 0..k, producing a predictive prior
    that cannot peek into future acoustic evidence.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        conv_kernel: int,
        dropout: float,
        max_slots: int = 200,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(max_slots, hidden_dim) * 0.02)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(hidden_dim, n_heads, ff_dim, conv_kernel, dropout, causal=True)
                for _ in range(n_layers)
            ]
        )

    def _get_pos(self, K: int) -> torch.Tensor:
        if K <= self.pos_embed.shape[0]:
            return self.pos_embed[:K]
        extra = torch.randn(
            K - self.pos_embed.shape[0],
            self.pos_embed.shape[1],
            device=self.pos_embed.device,
            dtype=self.pos_embed.dtype,
        ) * 0.02
        return torch.cat([self.pos_embed, extra], dim=0)

    def forward(self, slots: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """slots: (B, K, H), slot_mask: (B, K) with 1 for valid. Returns (B, K, H)."""
        K = slots.shape[1]
        z = self.input_norm(slots + self._get_pos(K))
        pad_mask = slot_mask < 0.5
        for layer in self.layers:
            z = layer(z, key_padding_mask=pad_mask)
        return z


def compute_sigreg_loss(
    z: torch.Tensor,
    slot_mask: torch.Tensor,
    n_projections: int = 64,
    n_t: int = 32,
) -> torch.Tensor:
    """SIGReg — Sketched-Isotropic-Gaussian Regularizer (Maes et al., LeWM 2026).

    Projects embeddings onto M random directions and minimises the Epps-Pulley
    test statistic along each 1-D projection to enforce an isotropic Gaussian
    latent distribution.  Single hyperparameter (weight λ), provable
    anti-collapse guarantee, smoother convergence than VICReg.

    Args:
        z:             (B, K, H) — beliefs or encoder output.
        slot_mask:     (B, K) float — 1 for valid slots.
        n_projections: M random projection directions (default 64).
        n_t:           Evaluation points for characteristic function (default 32).

    Returns:
        Scalar regularisation loss.
    """
    mask = slot_mask > 0.5
    z_flat = z[mask].float()
    N, H = z_flat.shape
    if N < 4:
        return z_flat.new_zeros(())

    u = F.normalize(
        torch.randn(H, n_projections, device=z_flat.device, dtype=z_flat.dtype), dim=0
    )
    h = z_flat @ u  # (N, M)

    h = (h - h.mean(0)) / h.std(0).clamp(min=1e-6)

    t = torch.linspace(0.5, 3.0, n_t, device=z_flat.device, dtype=z_flat.dtype)
    th = h.unsqueeze(-1) * t
    ecf_real = th.cos().mean(0)
    ecf_imag = th.sin().mean(0)
    gcf = (-0.5 * t.pow(2)).exp()

    return ((ecf_real - gcf) ** 2 + ecf_imag ** 2).mean()
