"""
Losses: SI-SNR implementation, L_f (feature disruption), L_s (separation suppression).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.critics import BSSCritic


def si_snr(source: torch.Tensor, estimate: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio (linear scale, differentiable).
    Higher is better; in L_s we minimize it to disrupt BSS.
    """
    source = source - source.mean(dim=dim, keepdim=True)
    estimate = estimate - estimate.mean(dim=dim, keepdim=True)
    s_dot_hat = (source * estimate).sum(dim=dim, keepdim=True)
    s_norm_sq = (source ** 2).sum(dim=dim, keepdim=True) + eps
    s_target = (s_dot_hat * source) / s_norm_sq
    e_noise = estimate - s_target
    s_target_sq = (s_target ** 2).sum(dim=dim)
    e_noise_sq = (e_noise ** 2).sum(dim=dim) + eps
    return s_target_sq / e_noise_sq


def si_snr_db(source: torch.Tensor, estimate: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """SI-SNR in dB for logging."""
    return 10 * torch.log10(si_snr(source, estimate, dim=dim, eps=eps) + eps)


def feature_disruption_loss(enc_clean: torch.Tensor, enc_jammed: torch.Tensor) -> torch.Tensor:
    """L_f: minimize cosine similarity between Enc(V_att) and Enc(V_att + S_probe)."""
    cos_sim = F.cosine_similarity(enc_clean.unsqueeze(0), enc_jammed.unsqueeze(1), dim=2).diag()
    return cos_sim.mean()


def separation_suppression_loss(v_att: torch.Tensor, mixture: torch.Tensor, bss: BSSCritic) -> torch.Tensor:
    """L_s: minimize SI-SNR between V_att and best-matching source in BSS estimate."""
    est = bss(mixture)
    if est.shape[1] < est.shape[2]:
        est = est.transpose(1, 2)
    B, n_src, T = est.shape
    v_att = v_att.to(est.device)
    if v_att.shape[-1] != T:
        v_att = F.pad(v_att, (0, max(0, T - v_att.shape[-1])))[..., :T]
    si_snr_per_src = [si_snr(v_att, est[:, k, :], dim=-1) for k in range(n_src)]
    stack = torch.stack(si_snr_per_src, dim=1)
    best_si_snr = stack.max(dim=1).values
    return best_si_snr.mean()
