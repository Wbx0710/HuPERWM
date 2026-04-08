"""Vocoder backends for mel-to-waveform conversion.

Provides two vocoders:
1. **HiFiGANVocoder** — bundled HiFi-GAN V1 generator that loads pre-trained
   weights from the official checkpoint format.
2. **GriffinLimVocoder** — torchaudio-based Griffin-Lim (no training required).

HiFi-GAN pre-trained weights
=============================
Download the official UNIVERSAL_V1 generator checkpoint::

    pip install gdown
    mkdir -p pretrained/hifigan
    gdown 1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW -O pretrained/hifigan/g_02500000

The checkpoint contains a dict with key ``"generator"`` holding the state dict.
The UNIVERSAL_V1 model was trained at 22 050 Hz but works well for 16 kHz mel
(same 0–8 kHz range and 80 mel bins; total upsample ratio = 256 = hop_length).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

LRELU_SLOPE = 0.1


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


# ---------------------------------------------------------------------------
# HiFi-GAN ResBlock (type 1)
# ---------------------------------------------------------------------------


class _ResBlock1(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=_get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


# ---------------------------------------------------------------------------
# HiFi-GAN V1 Generator
# ---------------------------------------------------------------------------


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN V1 (UNIVERSAL) generator for mel → waveform.

    Architecture matches the official config:
        upsample_rates          = [8, 8, 2, 2]          (total = 256 = hop_length)
        upsample_kernel_sizes   = [16, 16, 4, 4]
        upsample_initial_channel = 512
        resblock_kernel_sizes   = [3, 7, 11]
        resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
    """

    UPSAMPLE_RATES = [8, 8, 2, 2]
    UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
    UPSAMPLE_INITIAL_CH = 512
    RESBLOCK_KERNEL_SIZES = [3, 7, 11]
    RESBLOCK_DILATION_SIZES = [(1, 3, 5), (1, 3, 5), (1, 3, 5)]

    def __init__(self, n_mels: int = 80) -> None:
        super().__init__()
        h_u = self.UPSAMPLE_INITIAL_CH
        num_kernels = len(self.RESBLOCK_KERNEL_SIZES)

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(n_mels, h_u, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(
            zip(self.UPSAMPLE_RATES, self.UPSAMPLE_KERNEL_SIZES)
        ):
            ch_in = h_u // (2**i)
            ch_out = h_u // (2 ** (i + 1))
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(ch_in, ch_out, k, u, padding=(k - u) // 2)
                )
            )
            for k_r, d_r in zip(
                self.RESBLOCK_KERNEL_SIZES, self.RESBLOCK_DILATION_SIZES
            ):
                self.resblocks.append(_ResBlock1(ch_out, k_r, d_r))

        ch_final = h_u // (2 ** len(self.UPSAMPLE_RATES))
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(ch_final, 1, 7, 1, padding=3)
        )

        self.num_kernels = num_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_mels, T_mel) → waveform: (B, 1, T_mel * 256)."""
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)
        for up in self.ups:
            nn.utils.remove_weight_norm(up)
        for block in self.resblocks:
            for c in block.convs1:
                nn.utils.remove_weight_norm(c)
            for c in block.convs2:
                nn.utils.remove_weight_norm(c)


# ---------------------------------------------------------------------------
# Vocoder wrappers with a common interface
# ---------------------------------------------------------------------------


class GriffinLimVocoder:
    """Log-mel → waveform via InverseMelScale + Griffin-Lim."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        sr: int = 16000,
        n_iter: int = 100,
    ) -> None:
        self.hop_length = hop_length
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sr,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
            power=1.0,
        )

    @torch.no_grad()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel: (T, n_mels) → waveform: (N,)."""
        mel_mag = log_mel.exp()  # undo log
        mel_mag = mel_mag.T  # (n_mels, T)
        linear = self.inv_mel(mel_mag)  # (n_stft, T)
        wav = self.griffin_lim(linear)  # (N,)
        peak = wav.abs().max().clamp(min=1e-5)
        return wav / peak * 0.95


class HiFiGANVocoder:
    """Log-mel → waveform via pre-trained HiFi-GAN generator."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = HiFiGANGenerator()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("generator", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.remove_weight_norm()
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel: (T, n_mels) → waveform: (N,)."""
        mel = log_mel.to(self.device).T.unsqueeze(0)  # (1, n_mels, T)
        wav = self.model(mel).squeeze(0).squeeze(0).cpu()  # (N,)
        peak = wav.abs().max().clamp(min=1e-5)
        return wav / peak * 0.95


def load_vocoder(
    vocoder_path: str | None = None,
    device: str = "cpu",
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    sr: int = 16000,
):
    """Load the best available vocoder.

    If *vocoder_path* points to a HiFi-GAN checkpoint, uses that.
    Otherwise falls back to Griffin-Lim.
    """
    if vocoder_path and Path(vocoder_path).is_file():
        print(f"  Vocoder: HiFi-GAN from {vocoder_path}")
        return HiFiGANVocoder(vocoder_path, device=device)
    if vocoder_path:
        print(f"  Warning: vocoder path '{vocoder_path}' not found, using Griffin-Lim")
    else:
        print("  Vocoder: Griffin-Lim (pass --vocoder-path for HiFi-GAN)")
    return GriffinLimVocoder(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sr)
