"""TTS decoder for the Identity-Aware Perceptual Speech World Model.

Implements a conditional flow-matching (OT-CFM / rectified-flow) decoder that
generates mel spectrograms from the shared belief state, language prior, and
identity embedding.  This validates the bidirectional hypothesis: the same
latent dynamics that explain recognition should also support production.

Architecture (v2 improvements over v1)
=======================================
- Mel normalization with pre-computed global mean/std
- Log-domain duration prediction (more stable, avoids systematic under-prediction)
- Frame-level sinusoidal positional encoding in the denoiser
- 1D-conv condition smoother after slot-to-frame upsampling
- Magnitude spectrum for mel (matches HiFi-GAN convention)

Training loss:  L_tts = E_{t,z}[ ||v_θ(x_t, t, c) - (x_1 - x_0)||² ]
    where x_0 ~ N(0,I), x_1 = normalized target mel, x_t = (1-t)x_0 + t·x_1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import librosa
import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TTSConfig:
    mel_dim: int = 80
    mel_hop: int = 256
    mel_sr: int = 16000
    mel_n_fft: int = 1024
    mel_f_min: float = 0.0
    mel_f_max: float = 8000.0
    cond_dim: int = 256
    identity_dim: int = 128
    decoder_dim: int = 512
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_ff_dim: int = 2048
    dropout: float = 0.1
    max_mel_len: int = 2000
    duration_predictor_dim: int = 256
    cond_smoother_kernel: int = 5


# ---------------------------------------------------------------------------
# Mel spectrogram extraction  (magnitude spectrum — matches HiFi-GAN)
# ---------------------------------------------------------------------------


class MelExtractor(nn.Module):
    """Differentiable log-mel spectrogram via torch.stft + mel filterbank.

    Uses **magnitude** spectrum (not power) and **Slaney** mel scale so that
    log-mel values are directly compatible with HiFi-GAN vocoders (which are
    trained with librosa's default Slaney mel filterbank).
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        mel_fb = torch.from_numpy(
            librosa.filters.mel(
                sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max,
            ).copy()
        ).float()  # (n_mels, n_fft//2+1) — identical to HiFi-GAN training
        self.register_buffer("mel_fb", mel_fb)
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, N) → log-mel: (B, T_mel, n_mels)."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.window,  # type: ignore[arg-type]
            return_complex=True,
        )
        mag = spec.abs()  # magnitude spectrum (NOT power — matches HiFi-GAN)
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)
        log_mel = torch.log(mel.clamp(min=1e-5))
        return log_mel.transpose(1, 2)  # (B, T_mel, n_mels)


# ---------------------------------------------------------------------------
# Mel normalizer  (global mean/std)
# ---------------------------------------------------------------------------


class MelNormalizer(nn.Module):
    """Per-bin mel normalization using pre-computed global statistics.

    Normalizing the mel target to roughly zero-mean unit-variance makes the
    flow-matching velocity field much easier to learn (the interpolation path
    stays centred around zero instead of drifting to −3.5).
    """

    def __init__(self, mel_dim: int = 80) -> None:
        super().__init__()
        self.register_buffer("mel_mean", torch.zeros(mel_dim))
        self.register_buffer("mel_std", torch.ones(mel_dim))
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_stats(self, stats_path: str) -> None:
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        self.mel_mean.copy_(stats["mean"])
        self.mel_std.copy_(stats["std"].clamp(min=1e-5))
        self._loaded = True

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mel_mean) / self.mel_std

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        return mel * self.mel_std + self.mel_mean


# ---------------------------------------------------------------------------
# Log-domain duration predictor
# ---------------------------------------------------------------------------


class DurationPredictor(nn.Module):
    """Predicts mel-frame durations per syllable slot in log-domain.

    Training in log-domain prevents the systematic under-prediction that
    occurs with Softplus + raw-MSE, because the loss penalises relative
    errors rather than absolute errors on large-valued durations.
    """

    def __init__(self, cond_dim: int, identity_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = cond_dim * 2 + identity_dim  # belief + prior + identity
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        beliefs: torch.Tensor,
        priors: torch.Tensor,
        identity: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (dur_frames, log_dur) each (B, K)."""
        B, K, _ = beliefs.shape
        i_exp = identity.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([beliefs, priors, i_exp], dim=-1)
        log_dur = self.net(x).squeeze(-1)
        dur = log_dur.exp().clamp(min=1.0) * slot_mask
        return dur, log_dur


# ---------------------------------------------------------------------------
# Positional encoding helpers
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map scalar time t ∈ [0,1] to *dim*-dimensional sinusoidal embedding."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = t.unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding table for mel frames."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Condition smoother  (1-D conv after upsampling)
# ---------------------------------------------------------------------------


class ConditionSmoother(nn.Module):
    """Smooths hard slot-boundary transitions in the upsampled condition.

    After slot-to-frame upsampling each slot's vector is simply repeated,
    creating sharp discontinuities at slot boundaries.  A small residual
    1-D convolution learns to smooth these into gradual transitions that
    better reflect natural coarticulation.
    """

    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → smoothed (B, T, D)."""
        residual = x
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + residual)


# ---------------------------------------------------------------------------
# Transformer denoiser block
# ---------------------------------------------------------------------------


class DenoiserBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.time_mlp(t_emb).unsqueeze(1)
        h = self.norm1(x)
        h = x + self.self_attn(h, h, h, key_padding_mask=mask)[0]
        h2 = self.norm2(h)
        h = h + self.cross_attn(h2, cond, cond)[0]
        h = h + self.ffn(self.norm3(h))
        return h


# ---------------------------------------------------------------------------
# Flow-Matching TTS Decoder  (v2)
# ---------------------------------------------------------------------------


class FlowMatchingTTSDecoder(nn.Module):
    """Conditional flow-matching decoder: noise → mel conditioned on belief states.

    v2 improvements:
    - MelNormalizer for stable flow-matching training
    - Log-domain DurationPredictor
    - Sinusoidal positional encoding over mel frames
    - ConditionSmoother for smooth slot-to-frame transitions
    - duration_scale parameter at inference time
    """

    def __init__(self, cfg: TTSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.decoder_dim

        self.input_proj = nn.Linear(cfg.mel_dim, D)
        cond_in = cfg.cond_dim * 2 + cfg.identity_dim  # belief + prior + identity
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )
        self.time_embed_dim = D
        self.time_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        self.register_buffer(
            "pos_encoding", _build_sinusoidal_pe(cfg.max_mel_len, D)
        )

        self.cond_smoother = ConditionSmoother(D, cfg.cond_smoother_kernel)

        self.layers = nn.ModuleList(
            [
                DenoiserBlock(D, cfg.decoder_heads, cfg.decoder_ff_dim, cfg.dropout)
                for _ in range(cfg.decoder_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, cfg.mel_dim)

        self.duration_predictor = DurationPredictor(
            cfg.cond_dim, cfg.identity_dim, cfg.duration_predictor_dim
        )
        self.mel_extractor = MelExtractor(
            sr=cfg.mel_sr,
            hop_length=cfg.mel_hop,
            n_mels=cfg.mel_dim,
        )
        self.mel_normalizer = MelNormalizer(cfg.mel_dim)

    # ---- conditioning helpers -----------------------------------------------

    def _build_condition(
        self,
        beliefs: torch.Tensor,
        priors: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate and project conditioning to decoder dim. (B, K, D)."""
        B, K, _ = beliefs.shape
        i_exp = identity.unsqueeze(1).expand(-1, K, -1)
        raw = torch.cat([beliefs, priors, i_exp], dim=-1)
        return self.cond_proj(raw)

    @staticmethod
    def _upsample_condition(
        cond: torch.Tensor,
        durations: torch.Tensor,
        max_len: int,
        slot_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat each slot's condition for its predicted duration.

        Returns (upsampled (B, T, D), mel_mask (B, T)).
        """
        B, K, D = cond.shape
        dur_int = durations.round().long()
        if slot_mask is not None:
            dur_int = dur_int * slot_mask.long()
        dur_int = dur_int.clamp(min=0)
        out = cond.new_zeros(B, max_len, D)
        mask = cond.new_zeros(B, max_len)
        for b in range(B):
            pos = 0
            for k in range(K):
                d = dur_int[b, k].item()
                if d <= 0:
                    continue
                end = min(pos + d, max_len)
                if end <= pos:
                    break
                out[b, pos:end] = cond[b, k]
                mask[b, pos:end] = 1.0
                pos = end
        return out, mask

    @staticmethod
    def _close_boundary_gaps(
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
        num_frames: torch.Tensor,
    ) -> torch.Tensor:
        """Extend Sylber boundaries to cover the full utterance (no gaps).

        Sylber leaves gaps at utterance start/end and between some syllables.
        For TTS, durations must tile the entire mel so that the packed condition
        aligns with the mel target.  Leading silence is absorbed into the first
        slot, trailing silence into the last, and inter-syllable gaps are split
        at their midpoint between neighbouring slots.
        """
        B, K, _ = boundaries.shape
        closed = boundaries.clone()
        for b in range(B):
            n = int(slot_mask[b].sum().item())
            if n == 0:
                continue
            closed[b, 0, 0] = 0
            for k in range(n - 1):
                gap_s = closed[b, k, 1].item()
                gap_e = closed[b, k + 1, 0].item()
                if gap_e > gap_s:
                    mid = (gap_s + gap_e + 1) // 2
                    closed[b, k, 1] = mid
                    closed[b, k + 1, 0] = mid
            closed[b, n - 1, 1] = num_frames[b]
        return closed

    @staticmethod
    def _compute_gt_durations(
        boundaries: torch.Tensor,
        slot_mask: torch.Tensor,
        mel_lengths: torch.Tensor,
        num_frames: torch.Tensor,
    ) -> torch.Tensor:
        """Convert Sylber frame-boundaries to mel-frame durations (per-example).

        Boundaries are first gap-closed so that durations tile the full mel,
        preventing condition–target misalignment during flow-matching training.
        """
        B = boundaries.shape[0]
        closed = FlowMatchingTTSDecoder._close_boundary_gaps(
            boundaries, slot_mask, num_frames
        )
        ratios = mel_lengths.float() / num_frames.float().clamp(min=1)  # (B,)
        mel_bounds = (closed.float() * ratios.view(B, 1, 1)).long()
        durations = (mel_bounds[:, :, 1] - mel_bounds[:, :, 0]).float()
        durations = durations * slot_mask
        # Correct rounding residual so durations sum exactly to mel_lengths.
        for b in range(B):
            n = int(slot_mask[b].sum().item())
            if n == 0:
                continue
            residual = mel_lengths[b].item() - int(durations[b].sum().item())
            durations[b, n - 1] += residual
        return durations

    # ---- training -----------------------------------------------------------

    def compute_loss(
        self,
        beliefs: torch.Tensor,
        priors: torch.Tensor,
        identity: torch.Tensor,
        slot_mask: torch.Tensor,
        mel_target: torch.Tensor,
        mel_lengths: torch.Tensor,
        boundaries: torch.Tensor,
        num_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Flow-matching loss + log-domain duration loss."""
        B, T_mel, M = mel_target.shape

        max_len = self.cfg.max_mel_len
        if T_mel > max_len:
            mel_target = mel_target[:, :max_len, :]
            mel_lengths = mel_lengths.clamp(max=max_len)
            T_mel = max_len

        mel_target = self.mel_normalizer.normalize(mel_target)

        cond = self._build_condition(beliefs, priors, identity)

        gt_dur = self._compute_gt_durations(
            boundaries, slot_mask, mel_lengths, num_frames
        )
        pred_dur, pred_log_dur = self.duration_predictor(
            beliefs, priors, identity, slot_mask
        )

        gt_log_dur = torch.log(gt_dur.clamp(min=1.0))
        dur_loss = F.mse_loss(
            pred_log_dur * slot_mask, gt_log_dur * slot_mask, reduction="none"
        )
        dur_loss = (dur_loss * slot_mask).sum() / slot_mask.sum().clamp_min(1.0)

        up_cond, mel_mask = self._upsample_condition(cond, gt_dur, T_mel, slot_mask)
        up_cond = self.cond_smoother(up_cond)

        length_mask = (
            torch.arange(T_mel, device=beliefs.device).unsqueeze(0)
            < mel_lengths.unsqueeze(1)
        )
        mel_mask = mel_mask * length_mask.float()

        t = torch.rand(B, device=beliefs.device)
        z = torch.randn_like(mel_target)
        t_bc = t.view(B, 1, 1)
        x_t = (1 - t_bc) * z + t_bc * mel_target
        v_target = mel_target - z

        t_emb = self.time_proj(sinusoidal_embedding(t, self.time_embed_dim))
        h = self.input_proj(x_t) + self.pos_encoding[:T_mel].unsqueeze(0)
        pad_mask = mel_mask < 0.5
        for layer in self.layers:
            h = layer(h, up_cond, t_emb, mask=pad_mask)
        v_pred = self.out_proj(self.out_norm(h))

        mel_mask_3d = mel_mask.unsqueeze(-1)
        n_mel_elements = mel_mask.sum().clamp_min(1.0) * v_pred.shape[-1]
        flow_loss = (
            F.mse_loss(
                v_pred * mel_mask_3d, v_target * mel_mask_3d, reduction="sum"
            )
            / n_mel_elements
        )

        return {
            "tts_flow_loss": flow_loss,
            "tts_dur_loss": dur_loss,
            "tts_loss": flow_loss + dur_loss,
        }

    # ---- inference ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        beliefs: torch.Tensor,
        priors: torch.Tensor,
        identity: torch.Tensor,
        slot_mask: torch.Tensor,
        n_steps: int = 32,
        temperature: float = 0.8,
        duration_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate mel spectrogram via Euler ODE integration.

        Returns (mel (B, T, M), mel_lengths (B,)).
        The returned mel is in **original** (denormalized) log-mel space.
        """
        B = beliefs.shape[0]
        cond = self._build_condition(beliefs, priors, identity)
        pred_dur, _ = self.duration_predictor(beliefs, priors, identity, slot_mask)
        pred_dur = pred_dur * duration_scale
        total_dur = (pred_dur * slot_mask).sum(dim=1).long()
        T_mel = total_dur.max().item()
        T_mel = min(int(T_mel), self.cfg.max_mel_len)

        up_cond, mel_mask = self._upsample_condition(
            cond, pred_dur, T_mel, slot_mask
        )
        up_cond = self.cond_smoother(up_cond)

        x = (
            torch.randn(B, T_mel, self.cfg.mel_dim, device=beliefs.device)
            * temperature
        )
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=beliefs.device)
            t_emb = self.time_proj(sinusoidal_embedding(t, self.time_embed_dim))
            h = self.input_proj(x) + self.pos_encoding[:T_mel].unsqueeze(0)
            pad_mask = mel_mask < 0.5
            for layer in self.layers:
                h = layer(h, up_cond, t_emb, mask=pad_mask)
            v = self.out_proj(self.out_norm(h))
            x = x + v * dt

        x = self.mel_normalizer.denormalize(x)
        return x, total_dur.clamp(max=T_mel)
