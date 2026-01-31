"""
LibriSpeech dataset and pool construction: fixed-length crop/pad, pool building, attacker DataLoader.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset

from config import AUDIO_LENGTH, LIBRISPEECH_URL, SAMPLE_RATE


class LibriSpeechCrop(Dataset):
    """LibriSpeech fixed-length crop/pad, outputs (length,) waveform."""

    def __init__(
        self,
        root: str,
        url: str = LIBRISPEECH_URL,
        download: bool = True,
        length: int = AUDIO_LENGTH,
    ):
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)
        self.length = length

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav, sr, *_ = self.ds[idx]
        assert sr == SAMPLE_RATE
        T = wav.shape[-1]
        if T >= self.length:
            start = torch.randint(0, T - self.length + 1, (1,)).item()
            return wav[..., start : start + self.length].squeeze(0)
        pad = self.length - T
        return F.pad(wav.squeeze(0), (0, pad), value=0.0)


def build_pool(dataset: LibriSpeechCrop, pool_size: int, device: torch.device) -> torch.Tensor:
    """Build fixed pool from first pool_size samples of dataset, shape (pool_size, audio_length)."""
    pool = []
    for i in range(min(pool_size, len(dataset))):
        pool.append(dataset[i])
    if len(pool) < pool_size:
        raise RuntimeError(
            f"LibriSpeech has only {len(pool)} samples, need at least {pool_size}. "
            "Use a larger subset (e.g. train-clean-360) or reduce pool_size."
        )
    return torch.stack(pool[:pool_size]).to(device)


def get_attacker_loader(
    dataset: LibriSpeechCrop,
    pool_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Sample only from indices >= pool_size so attacker speech does not overlap with pool."""
    if len(dataset) <= pool_size:
        raise RuntimeError("Dataset too small; cannot draw attacker samples beyond the pool.")
    subset = Subset(dataset, list(range(pool_size, len(dataset))))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
