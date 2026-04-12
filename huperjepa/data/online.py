"""Online feature extraction from HuggingFace LibriSpeech.

Loads audio on-the-fly via HuggingFace Datasets and extracts HuPER hidden
states + Sylber syllable boundaries inside ``__getitem__``.
"""

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from g2p_en import G2p
from transformers import Wav2Vec2Processor, WavLMForCTC

from huperjepa.data.vocab import MIN_WAVLM_INPUT_SAMPLES, Vocabulary, ensure_min_audio_length


def _extract_sylber_boundaries(segmenter, audio: torch.Tensor) -> torch.Tensor:
    wav = audio.unsqueeze(0) if audio.dim() == 1 else audio
    std = wav.std()
    if std < 1e-6:
        std = torch.tensor(1.0, device=wav.device)
    wav = (wav - wav.mean()) / std
    result = segmenter(wav=[wav], in_second=False)
    segments = result[0]["segments"]
    if len(segments) == 0:
        num_frames = max(1, audio.shape[-1] // 320)
        return torch.tensor([[0, num_frames]], dtype=torch.long)
    return torch.from_numpy(np.asarray(segments, dtype=np.int64))


class OnlineLibriSpeechWMDataset(torch.utils.data.Dataset):
    """LibriSpeech with on-the-fly HuPER + Sylber feature extraction.

    Lazy-initialises models on first ``__getitem__`` call so the dataset can
    be constructed in the main process and used in DataLoader workers.
    """

    def __init__(
        self,
        hf_dataset_name: str,
        hf_dataset_config: str,
        hf_split: str,
        phone_vocab: Vocabulary,
        text_vocab: Vocabulary,
        evidence_type: str = "hidden",  # "hidden" (1024-dim) recommended
        huper_repo: str = "huper29/huper_recognizer",
        feature_device: str = "cuda",
        max_examples: int | None = None,
        teacher_cache: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.ds = load_dataset(hf_dataset_name, hf_dataset_config, split=hf_split)
        self.ds = self.ds.cast_column("audio", Audio(decode=False))
        if max_examples is not None:
            self.ds = self.ds.select(range(min(len(self.ds), max_examples)))

        self.phone_vocab = phone_vocab
        self.text_vocab = text_vocab
        self.evidence_type = evidence_type
        self.huper_repo = huper_repo
        self.feature_device = feature_device
        self.teacher_cache = teacher_cache

        # Lazy-init on first __getitem__
        self._processor = None
        self._huper = None
        self._segmenter = None
        self._g2p = None

    def __len__(self) -> int:
        return len(self.ds)

    def _lazy_init(self) -> None:
        if self._processor is not None:
            return
        device = torch.device(self.feature_device if torch.cuda.is_available() else "cpu")
        self._processor = Wav2Vec2Processor.from_pretrained(self.huper_repo)
        self._huper = WavLMForCTC.from_pretrained(self.huper_repo).to(device).eval()
        from sylber import Segmenter
        self._segmenter = Segmenter(device=str(device))
        self._g2p = G2p()

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict:
        self._lazy_init()
        ex = self.ds[idx]
        seg_id = ex.get("id", str(idx))
        text = ex.get("text", "")
        text_chars = list(text)

        audio_bytes = ex["audio"]["bytes"]
        audio_path = ex["audio"].get("path")
        if audio_bytes is not None:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        elif audio_path is not None:
            audio_np, sr = sf.read(audio_path, dtype="float32")
        else:
            raise ValueError(f"No audio data for example {idx}")

        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        if sr != 16000:
            import torchaudio
            audio_t = torch.from_numpy(audio_np).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            audio_np = audio_t.squeeze(0).numpy()

        audio = torch.from_numpy(audio_np).float()
        audio = ensure_min_audio_length(audio, MIN_WAVLM_INPUT_SAMPLES)
        device = next(self._huper.parameters()).device  # type: ignore[union-attr]

        inputs = self._processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        need_hidden = self.evidence_type == "hidden"
        out = self._huper(**inputs, output_hidden_states=need_hidden)
        if need_hidden:
            evidence = out.hidden_states[-1][0].float().cpu()
        else:
            evidence = out.logits[0].float().cpu()

        boundaries = _extract_sylber_boundaries(self._segmenter, audio.to(device)).cpu()
        num_frames = int(evidence.shape[0])
        boundaries = boundaries.clamp(min=0, max=num_frames)
        valid = boundaries[:, 1] > boundaries[:, 0]
        boundaries = boundaries[valid]
        if boundaries.shape[0] == 0:
            boundaries = torch.tensor([[0, num_frames]], dtype=torch.long)

        phones_raw = self._g2p(text)
        canonical = [p for p in phones_raw if p.strip() and not re.match(r"^[^\w]+$", p)]
        if self.teacher_cache is not None and seg_id in self.teacher_cache:
            teacher = list(self.teacher_cache[seg_id])
        else:
            teacher = list(canonical)

        return {
            "segment_id": seg_id,
            "evidence": evidence,
            "num_frames": evidence.shape[0],
            "boundaries": boundaries,
            "num_syllables": boundaries.shape[0],
            "text": text,
            "text_ids": torch.tensor(self.text_vocab.encode(text_chars), dtype=torch.long),
            "canonical_phones": canonical,
            "canonical_ids": torch.tensor(self.phone_vocab.encode(canonical), dtype=torch.long),
            "teacher_phones": teacher,
            "teacher_ids": torch.tensor(self.phone_vocab.encode(teacher), dtype=torch.long),
        }
