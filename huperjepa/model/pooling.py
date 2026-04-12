"""Syllable-slot pooling module.

Provides EnhancedBoundaryAttnPool: boundary-constrained cross-attention that
pools 50 Hz HuPER frame features into Sylber syllable slots, followed by a
causal self-attention pass for inter-slot contextual refinement.

This is the only pooling module used in v3 — simpler alternatives (mean,
energy-weighted, plain boundary attention) have been removed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_pool_frames(
    projected: torch.Tensor,
    boundaries: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool frame features into syllable slots (used internally as init query)."""
    B, T, H = projected.shape
    K = slot_mask.shape[1]
    tidx = torch.arange(T, device=projected.device).view(1, T, 1)
    starts = boundaries[:, :, 0].unsqueeze(1)
    ends = boundaries[:, :, 1].unsqueeze(1)
    in_slot = (tidx >= starts) & (tidx < ends) & (slot_mask.unsqueeze(1) > 0)
    w = in_slot.float()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.einsum("btk,bth->bkh", w, projected)


def _boundary_attn_mask(
    boundaries: torch.Tensor,
    slot_mask: torch.Tensor,
    T: int,
    num_heads: int,
) -> torch.Tensor:
    """Build (B*num_heads, K, T) boolean attention mask.

    True = position is blocked (outside the syllable boundary or padding slot).
    """
    B, K = slot_mask.shape
    tidx = torch.arange(T, device=boundaries.device).view(1, 1, T)
    starts = boundaries[:, :, 0].unsqueeze(-1)
    ends = boundaries[:, :, 1].unsqueeze(-1)
    in_boundary = (tidx >= starts) & (tidx < ends)
    valid_slot = (slot_mask > 0.5).unsqueeze(-1)
    blocked = ~(in_boundary & valid_slot)
    return blocked.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(B * num_heads, K, T)


class EnhancedBoundaryAttnPool(nn.Module):
    """Boundary-gated cross-attention pooling with causal inter-slot refinement.

    Step 1 — Frame aggregation: each slot query (initialised from mean-pooled
    frames) cross-attends only to the HuPER frames within its Sylber boundary.
    This preserves sub-syllabic acoustic trajectories (formant transitions,
    closure bursts) discarded by plain mean pooling.

    Step 2 — Slot contextualisation: a causal self-attention pass lets each
    slot adjust its representation based on preceding slots, mirroring the
    top-down temporal flow of linguistic prediction.

    Forward signature: (projected, boundaries, slot_mask) → (B, K, H).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        projected: torch.Tensor,   # (B, T, H)
        boundaries: torch.Tensor,  # (B, K, 2)
        slot_mask: torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        B, T, H = projected.shape
        K = slot_mask.shape[1]

        # Step 1: boundary-gated cross-attention.
        init_slots = _mean_pool_frames(projected, boundaries, slot_mask)
        queries = self.query_proj(init_slots)
        boundary_mask = _boundary_attn_mask(boundaries, slot_mask, T, self.num_heads)
        attn_out, _ = self.cross_attn(
            query=queries,
            key=projected,
            value=projected,
            attn_mask=boundary_mask,
            need_weights=False,
        )
        slots = self.cross_norm(attn_out + queries)

        # Step 2: causal self-attention for inter-slot context.
        pad_mask = slot_mask < 0.5
        causal_mask = torch.triu(
            torch.full((K, K), float("-inf"), device=slots.device), diagonal=1
        )
        ctx, _ = self.self_attn(
            slots, slots, slots,
            key_padding_mask=pad_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        return self.output_norm(ctx + slots)
