"""JEPA components for the Belief World Model.

Implements the Joint-Embedding Predictive Architecture adapted to syllable-slot
level speech representations, following Ioannides et al. (2025) "JEPA as a
Neural Tokenizer".

Components
==========
1.  DensityAdaptiveAttention (DAAM) — Gaussian mixture gating for adaptive
    temporal feature selection (paper §2.2).
2.  ConformerBlock — Self-attention + depthwise conv + FFN with DAAM gating.
3.  ConformerDAAMEncoder — Stack of ConformerBlocks (online / target encoder).
4.  JEPAPredictor — Small Conformer that predicts masked slot representations.
5.  block_mask_slots — Contiguous block masking at the syllable-slot level.
6.  CausalConformerEncoder — Unidirectional Conformer for the language prior L_t.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JEPAConfig:
    hidden_dim: int = 256
    encoder_layers: int = 6
    encoder_heads: int = 8
    encoder_ff_dim: int = 1024
    encoder_conv_kernel: int = 15
    daam_num_gaussians: int = 4
    daam_alpha_init: float = 0.05
    predictor_layers: int = 2
    predictor_heads: int = 8
    prior_layers: int = 2
    prior_heads: int = 8
    dropout: float = 0.1
    ema_tau: float = 0.996
    mask_ratio: float = 0.5
    mask_min_span: int = 1
    mask_max_span: int | None = None  # defaults to K // 4
    max_slots: int = 200


# ---------------------------------------------------------------------------
# Density Adaptive Attention (DAAM)  — paper §2.2
# ---------------------------------------------------------------------------


class DensityAdaptiveAttention(nn.Module):
    """Gaussian-mixture density gating over the slot (temporal) axis.

    For input x ∈ (B, K, H) the module computes a 1-channel attention
    projection, evaluates K_gauss Gaussian log-densities, and produces a
    multiplicative gate via logsumexp aggregation.

    All critical computations run in FP32 for numerical stability.
    """

    def __init__(
        self, hidden_dim: int, num_gaussians: int = 4, alpha_init: float = 0.05
    ) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians

        self.proj = nn.Conv1d(hidden_dim, 1, 1)
        self.delta = nn.Parameter(torch.zeros(num_gaussians))
        self.log_scale = nn.Parameter(
            torch.full((num_gaussians,), math.log(0.5))
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, K, H), mask: (B, K) with 1 for valid positions."""
        orig_dtype = x.dtype
        x_fp32 = x.float()
        B, K, H = x_fp32.shape

        proj_1d = self.proj(x_fp32.transpose(1, 2)).squeeze(1)  # (B, K)

        if mask is not None:
            proj_masked = proj_1d.masked_fill(mask < 0.5, 0.0)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            proj_masked = proj_1d
            counts = torch.full((B, 1), float(K), device=x.device)

        mu = proj_masked.sum(dim=1, keepdim=True) / counts  # (B, 1)
        diff = proj_masked - mu
        if mask is not None:
            diff = diff * mask
        var = (diff ** 2).sum(dim=1, keepdim=True) / counts  # (B, 1)
        var = var.clamp(min=1e-6)

        sigma = F.softplus(self.log_scale) + 1e-3  # (K_gauss,)
        log_gates = []

        for k in range(self.num_gaussians):
            z_k = (proj_1d - (mu + self.delta[k])) / (var.sqrt() * sigma[k] + 1e-3)
            log_p = -0.5 * z_k ** 2 - torch.log(sigma[k]) - 0.5 * math.log(2 * math.pi)
            log_gates.append(log_p)

        log_G = torch.logsumexp(torch.stack(log_gates, dim=0), dim=0) - math.log(
            self.num_gaussians
        )
        gate = log_G.exp().unsqueeze(-1)  # (B, K, 1)

        out = x_fp32 * (1.0 + self.alpha * gate)
        return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Conformer building blocks
# ---------------------------------------------------------------------------


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

    When *causal* is True the depthwise conv uses left-only padding so that
    position k sees only positions 0..k, matching the causal attention in
    the prior encoder.

    Uses GroupNorm (groups=32 or dim if dim<32) instead of BatchNorm1d so
    that normalisation statistics are independent of padding positions and
    batch composition—important for variable-length syllable-slot sequences.
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
            x = x.transpose(1, 2)                # (B, K, H)
            x = self.conv_norm(x)                 # per-position LayerNorm
            x = x.transpose(1, 2)                 # (B, H, K)
        else:
            x = self.conv_norm(x)                 # GroupNorm over (H, K)
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


# ---------------------------------------------------------------------------
# Conformer + DAAM Encoder  (used for both online and target)
# ---------------------------------------------------------------------------


class ConformerDAAMEncoder(nn.Module):
    """Stack of ConformerBlocks with DAAM gating after each block."""

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        H = cfg.hidden_dim
        self.pos_embed = nn.Parameter(torch.randn(cfg.max_slots, H) * 0.02)
        self.input_norm = nn.LayerNorm(H)
        self.output_norm = nn.LayerNorm(H)
        self.layers = nn.ModuleList()
        self.daam_gates = nn.ModuleList()
        for _ in range(cfg.encoder_layers):
            self.layers.append(
                ConformerBlock(
                    H,
                    cfg.encoder_heads,
                    cfg.encoder_ff_dim,
                    cfg.encoder_conv_kernel,
                    cfg.dropout,
                    causal=False,
                )
            )
            self.daam_gates.append(
                DensityAdaptiveAttention(H, cfg.daam_num_gaussians, cfg.daam_alpha_init)
            )

    def _get_pos(self, K: int) -> torch.Tensor:
        if K <= self.pos_embed.shape[0]:
            return self.pos_embed[:K]
        extra = torch.randn(
            K - self.pos_embed.shape[0], self.pos_embed.shape[1],
            device=self.pos_embed.device, dtype=self.pos_embed.dtype,
        ) * 0.02
        return torch.cat([self.pos_embed, extra], dim=0)

    def forward(
        self, slots: torch.Tensor, slot_mask: torch.Tensor
    ) -> torch.Tensor:
        """slots: (B, K, H), slot_mask: (B, K) → z: (B, K, H)."""
        K = slots.shape[1]
        z = self.input_norm(slots + self._get_pos(K))
        pad_mask = slot_mask < 0.5
        for layer, daam in zip(self.layers, self.daam_gates):
            z = layer(z, key_padding_mask=pad_mask)
            z = daam(z, mask=slot_mask)
        return self.output_norm(z)


# ---------------------------------------------------------------------------
# Causal Conformer Encoder  (for language prior L_t)
# ---------------------------------------------------------------------------


class CausalConformerEncoder(nn.Module):
    """Causal (unidirectional) Conformer for the language prior state.

    Each slot k only attends to slots 0..k, producing a proper predictive
    prior that does not peek into future acoustic evidence.
    """

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        H = cfg.hidden_dim
        self.pos_embed = nn.Parameter(torch.randn(cfg.max_slots, H) * 0.02)
        self.input_norm = nn.LayerNorm(H)
        self.layers = nn.ModuleList()
        for _ in range(cfg.prior_layers):
            self.layers.append(
                ConformerBlock(
                    H,
                    cfg.prior_heads,
                    cfg.encoder_ff_dim,
                    cfg.encoder_conv_kernel,
                    cfg.dropout,
                    causal=True,
                )
            )

    def _get_pos(self, K: int) -> torch.Tensor:
        if K <= self.pos_embed.shape[0]:
            return self.pos_embed[:K]
        extra = torch.randn(
            K - self.pos_embed.shape[0], self.pos_embed.shape[1],
            device=self.pos_embed.device, dtype=self.pos_embed.dtype,
        ) * 0.02
        return torch.cat([self.pos_embed, extra], dim=0)

    def forward(
        self, slots: torch.Tensor, slot_mask: torch.Tensor
    ) -> torch.Tensor:
        K = slots.shape[1]
        z = self.input_norm(slots + self._get_pos(K))
        pad_mask = slot_mask < 0.5
        for layer in self.layers:
            z = layer(z, key_padding_mask=pad_mask)
        return z


# ---------------------------------------------------------------------------
# JEPA Predictor Network  — paper §2.4
# ---------------------------------------------------------------------------


class JEPAPredictor(nn.Module):
    """Predicts target representations at masked positions.

    Uses a residual architecture: z_pred = LayerNorm(z_masked + delta).
    The predictor learns a *correction* rather than a full transformation,
    preserving encoder quality at unmasked positions (where the encoder
    already closely matches the target) while focusing capacity on
    predicting masked positions from context.

    The output projection is zero-initialized so the predictor starts as
    an identity function, avoiding early-training representation damage.
    """

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        H = cfg.hidden_dim
        H2 = H * 2
        self.expand = nn.Sequential(nn.Linear(H, H2), nn.GELU())
        self.layers = nn.ModuleList()
        for _ in range(cfg.predictor_layers):
            self.layers.append(
                ConformerBlock(
                    H2,
                    cfg.predictor_heads,
                    H2 * 2,
                    cfg.encoder_conv_kernel,
                    cfg.dropout,
                    causal=False,
                )
            )
        self.project = nn.Linear(H2, H)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.output_norm = nn.LayerNorm(H)

    def forward(
        self, z_masked: torch.Tensor, slot_mask: torch.Tensor
    ) -> torch.Tensor:
        """z_masked: (B, K, H) → z_pred: (B, K, H)."""
        h = self.expand(z_masked)
        pad_mask = slot_mask < 0.5
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad_mask)
        delta = self.project(h)
        return self.output_norm(z_masked + delta)


# ---------------------------------------------------------------------------
# Block Masking Strategy  — paper §2.1
# ---------------------------------------------------------------------------


def block_mask_slots(
    B: int,
    K: int,
    device: torch.device,
    mask_ratio: float = 0.5,
    min_span: int = 1,
    max_span: int | None = None,
) -> torch.Tensor:
    """Generate contiguous block masks for syllable slots.

    Returns a boolean tensor (B, K) where True = masked (target) position.

    Uses ``torch.randint`` so that masking is controlled by the PyTorch
    global random seed (set via ``pl.seed_everything``).
    """
    if max_span is None:
        max_span = max(1, K // 4)
    max_span = max(min_span, min(max_span, K))
    span_range = max_span - min_span + 1
    max_start = max(1, K - min_span + 1)

    mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    for b in range(B):
        n_target = max(1, int(mask_ratio * K))
        n_masked = 0
        attempts = 0
        while n_masked < n_target and attempts < 100:
            span = int(torch.randint(min_span, min_span + span_range, (1,)).item())
            start = int(torch.randint(0, max(1, K - span + 1), (1,)).item())
            end = min(start + span, K)
            mask[b, start:end] = True
            n_masked = int(mask[b].sum().item())
            attempts += 1
    return mask


# ---------------------------------------------------------------------------
# JEPA Loss
# ---------------------------------------------------------------------------


def compute_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    jepa_mask: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalized MSE loss on masked positions only (paper §2.5.1).

    L2-normalizes z_pred and z_target before MSE so the loss is
    scale-invariant, preventing representation expansion.

    z_pred, z_target: (B, K, H)
    jepa_mask: (B, K) bool — True = masked
    slot_mask: (B, K) float — 1.0 for valid slots
    """
    z_pred_n = F.normalize(z_pred.float(), dim=-1)
    z_target_n = F.normalize(z_target.float().detach(), dim=-1)

    valid_masked = jepa_mask & (slot_mask > 0.5)
    mask_3d = valid_masked.unsqueeze(-1).float()  # (B, K, 1)
    n_masked = mask_3d.sum().clamp(min=1.0)

    # ||norm(a) - norm(b)||^2 = 2 - 2*cos(a,b), bounded in [0, 4]
    loss = ((z_pred_n - z_target_n).pow(2) * mask_3d).sum() / n_masked
    return loss


# ---------------------------------------------------------------------------
# Variance-Covariance Regularization (VICReg-style)
# ---------------------------------------------------------------------------


def compute_vicreg_loss(
    z: torch.Tensor,
    slot_mask: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance + Covariance regularization to prevent representation collapse.

    Operates on L2-normalized representations (same space as the JEPA loss)
    to avoid gradient conflicts between magnitude-pushing VICReg and
    direction-only normalized MSE.

    Args:
        z: (B, K, H) — predictor output
        slot_mask: (B, K) — 1.0 for valid slots
        gamma: target standard deviation per dimension (variance hinge threshold)

    Returns:
        (var_loss, cov_loss) — both scalar tensors
    """
    mask = slot_mask > 0.5
    z_flat = F.normalize(z[mask].float(), dim=-1)  # (N, H)

    if z_flat.shape[0] < 2:
        zero = z.new_zeros(())
        return zero, zero

    std = z_flat.std(dim=0)  # (H,)
    # gamma scaled for unit-norm vectors: 1/sqrt(H) is the expected std
    # when directions are uniformly distributed; use gamma as a multiplier
    target_std = gamma / (z_flat.shape[-1] ** 0.5)
    var_loss = F.relu(target_std - std).mean()

    z_c = z_flat - z_flat.mean(dim=0)
    N = z_c.shape[0]
    cov = (z_c.T @ z_c) / max(N - 1, 1)  # (H, H)
    H = cov.shape[0]
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / max(H * (H - 1), 1)

    return var_loss, cov_loss


# ---------------------------------------------------------------------------
# EMA Update
# ---------------------------------------------------------------------------


@torch.no_grad()
def update_ema(
    online: nn.Module, target: nn.Module, tau: float = 0.996
) -> None:
    """Exponential moving average update: target ← τ·target + (1-τ)·online."""
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


# ---------------------------------------------------------------------------
# Collapse Monitor
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_sigreg_loss(
    z: torch.Tensor,
    slot_mask: torch.Tensor,
    n_projections: int = 64,
    n_t: int = 32,
) -> torch.Tensor:
    """SIGReg — Sketched-Isotropic-Gaussian Regularizer (Maes et al., LeWM 2026).

    Projects embeddings onto M random directions and minimises the Epps-Pulley
    test statistic along each 1-D projection to enforce an isotropic Gaussian
    latent distribution.  Replaces VICReg: single hyperparameter (weight λ),
    provable anti-collapse guarantee, smoother convergence.

    Args:
        z:            (B, K, H) — beliefs or predictor output.
        slot_mask:    (B, K) float — 1 for valid slots.
        n_projections: M random projection directions (default 64).
        n_t:          K evaluation points for the characteristic function (default 32).

    Returns:
        Scalar regularisation loss.
    """
    mask = slot_mask > 0.5
    z_flat = z[mask].float()          # (N, H)
    N, H = z_flat.shape
    if N < 4:
        return z_flat.new_zeros(())

    # --- random projections (new directions each forward pass for stochasticity) ---
    u = F.normalize(
        torch.randn(H, n_projections, device=z_flat.device, dtype=z_flat.dtype), dim=0
    )                                  # (H, M)
    h = z_flat @ u                     # (N, M)

    # standardise each projection → compare against N(0,1) target
    h = (h - h.mean(0)) / h.std(0).clamp(min=1e-6)   # (N, M)

    # Epps-Pulley: compare empirical CF to Gaussian CF at t grid
    # CF_N(0,1)(t) = exp(-t²/2)  [purely real, zero imaginary part]
    t = torch.linspace(0.5, 3.0, n_t, device=z_flat.device, dtype=z_flat.dtype)  # (K,)

    # h: (N, M) → unsqueeze for broadcasting with t: (K,)
    th = h.unsqueeze(-1) * t           # (N, M, K)
    ecf_real = th.cos().mean(0)        # (M, K)  — real part of empirical CF
    ecf_imag = th.sin().mean(0)        # (M, K)  — imaginary part
    gcf = (-0.5 * t.pow(2)).exp()      # (K,)    — Gaussian CF (real)

    return ((ecf_real - gcf) ** 2 + ecf_imag ** 2).mean()


def check_collapse(z_pred: torch.Tensor, threshold: float = 0.01) -> float:
    """Returns mean std of predictor output across batch and slot dims.

    If this falls below `threshold`, representation collapse is likely.
    """
    return z_pred.std(dim=[0, 1]).mean().item()
