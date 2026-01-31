"""
Fixed critics: HuBERT (content encoding), Conv-TasNet (BSS separation), both frozen.
"""
from __future__ import annotations

import torch
from transformers import HubertModel

try:
    from asteroid.models import ConvTasNet
except ImportError as e:
    _asteroid_err = e
    ConvTasNet = None  # type: ignore
else:
    _asteroid_err = None

from config import CONVTASNET_ID, HUBERT_ID


class HuBERTCritic(torch.nn.Module):
    """Frozen HuBERT: input waveform (B, T), output pooled hidden state (B, D); gradients flow through input."""

    def __init__(self, model_id: str = HUBERT_ID):
        super().__init__()
        self.model = HubertModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        wav = wav.to(device)
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-8)
        out = self.model(input_values=wav)
        return out.last_hidden_state.mean(1)


class BSSCritic(torch.nn.Module):
    """Frozen Conv-TasNet: input mixture (B, T), output separation (B, n_src, T) or (B, T, n_src)."""

    def __init__(self, model_id: str = CONVTASNET_ID):
        super().__init__()
        if ConvTasNet is None:
            msg = "Please install asteroid: pip install asteroid"
            if _asteroid_err is not None:
                msg += f"\nImport error: {_asteroid_err}"
            raise ImportError(msg)
        self.model = ConvTasNet.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        mixture = mixture.to(device)
        return self.model(mixture)
