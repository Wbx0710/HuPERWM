"""Vocabulary and shared text utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

PAD_TOKEN = "<pad>"
BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
MIN_WAVLM_INPUT_SAMPLES = 400


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def read_json(path) -> dict:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_jsonl(path) -> List[dict]:
    rows = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def ensure_min_audio_length(samples, min_length=MIN_WAVLM_INPUT_SAMPLES):
    if samples.dim() == 1:
        n = samples.shape[0]
        if n >= min_length:
            return samples
        return F.pad(samples, (0, min_length - n))
    n = samples.shape[-1]
    if n >= min_length:
        return samples
    return F.pad(samples, (0, min_length - n))


def normalize_text_for_char_ctc(text: str) -> List[str]:
    """Lowercase and keep [a-z], apostrophe and space only."""
    chars: List[str] = []
    prev_space = True
    for ch in text.lower():
        if "a" <= ch <= "z" or ch == "'":
            chars.append(ch)
            prev_space = False
        elif ch.isspace():
            if not prev_space:
                chars.append(" ")
            prev_space = True
    if chars and chars[-1] == " ":
        chars.pop()
    return chars


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@dataclass
class Vocabulary:
    tokens: List[str]

    def __post_init__(self):
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("Vocabulary contains duplicate tokens")
        for req in (PAD_TOKEN, BLANK_TOKEN, UNK_TOKEN):
            if req not in self.tokens:
                raise ValueError(f"Vocabulary must contain {req!r}")
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}

    @property
    def blank_id(self) -> int:
        return self.token_to_id[BLANK_TOKEN]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.unk_id
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode_ctc(self, ids) -> List[str]:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        blank = self.blank_id
        out: List[str] = []
        prev = None
        for idx in ids:
            if idx == blank:
                prev = None
                continue
            if idx == prev:
                continue
            out.append(self.tokens[idx])
            prev = idx
        return out

    def to_dict(self) -> dict:
        return {"tokens": self.tokens}

    @classmethod
    def from_file(cls, path):
        data = read_json(path)
        return cls(tokens=list(data["tokens"]))


def build_vocab(token_sequences: Iterable) -> Vocabulary:
    seen: set = set()
    for seq in token_sequences:
        for t in seq:
            seen.add(t)
    specials = {PAD_TOKEN, BLANK_TOKEN, UNK_TOKEN}
    content = sorted(t for t in seen if t not in specials)
    return Vocabulary(tokens=[PAD_TOKEN, BLANK_TOKEN, UNK_TOKEN] + content)


def levenshtein_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]
