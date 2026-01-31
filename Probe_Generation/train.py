"""
GiV training entry: assemble config, data, models and losses; run training and export probe.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

# When torchcodec is not installed, use soundfile as fallback for torchaudio.load (avoids TorchCodec is required error)
try:
    import torchcodec  # noqa: F401
except ImportError:
    _orig_load = torchaudio.load
    import soundfile as sf

    def _load(path, *args, **kwargs):
        try:
            return _orig_load(path, *args, **kwargs)
        except ImportError as e:
            if "torchcodec" in str(e).lower():
                wav, sr = sf.read(path, dtype="float32")
                wav = torch.from_numpy(wav).float()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                return wav, sr
            raise

    torchaudio.load = _load

from tqdm import tqdm

from config import (
    CONVTASNET_ID,
    HUBERT_ID,
    LIBRISPEECH_URL,
    SAMPLE_RATE,
    get_device,
    parse_args,
)
from data import LibriSpeechCrop, build_pool, get_attacker_loader
from losses import feature_disruption_loss, separation_suppression_loss
from models import BSSCritic, Generator, HuBERTCritic


def _device_str(device: torch.device) -> str:
    """Return human-readable device description: GPU model or CPU."""
    if device.type == "cuda":
        try:
            return f"GPU: {torch.cuda.get_device_name(device)}"
        except Exception:
            return "GPU"
    return "CPU"


def train_step(G, pool, hubert, bss, v_att, z, lambda_f, lambda_s, device):
    """One training step: forward, L_f + L_s, update G only."""
    G.train()
    pool = pool.to(device)
    v_att = v_att.to(device)
    z = z.to(device)
    w = G(z)
    s_probe = G.probe(w, pool)
    T = min(v_att.shape[-1], s_probe.shape[-1])
    v_att = v_att[..., :T]
    s_probe = s_probe[..., :T]
    jammed = v_att + s_probe

    enc_clean = hubert(v_att)
    enc_jammed = hubert(jammed)
    L_f = feature_disruption_loss(enc_clean, enc_jammed)
    L_s = separation_suppression_loss(v_att, jammed, bss)

    total = lambda_f * L_f + lambda_s * L_s
    return total, L_f.item(), L_s.item()


def main():
    args = parse_args()
    device = torch.device(get_device(args))
    torch.manual_seed(args.seed)

    print(f"Training device: {_device_str(device)}")
    if device.type == "cpu" and torch.cuda.is_available() is False:
        print("Note: PyTorch did not detect a GPU. If you have a GPU, install CUDA build: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("-" * 50)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / args.probe_path

    # Data
    print("Loading LibriSpeech...")
    libri = LibriSpeechCrop(
        root=args.librispeech_root,
        url=args.librispeech_url,
        download=True,
        length=args.audio_length,
    )
    pool = build_pool(libri, args.pool_size, device)
    print(f"Pool shape: {pool.shape}")

    attacker_loader = get_attacker_loader(
        libri, args.pool_size, args.batch_size, num_workers=args.num_workers
    )

    # Models
    G = Generator(latent_dim=args.latent_dim, pool_size=args.pool_size).to(device)
    hubert = HuBERTCritic(HUBERT_ID).to(device)
    bss = BSSCritic(CONVTASNET_ID).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=args.lr)

    # Training (with progress bar)
    for epoch in range(args.epochs):
        total_sum, lf_sum, ls_sum, n = 0.0, 0.0, 0.0, 0
        pbar = tqdm(
            attacker_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True,
            unit="batch",
            ncols=100,
        )
        for v_att_batch in pbar:
            v_att_batch = v_att_batch.to(device)
            B = v_att_batch.shape[0]
            z = torch.randn(B, args.latent_dim, device=device)
            loss, lf, ls = train_step(
                G, pool, hubert, bss, v_att_batch, z,
                args.lambda_f, args.lambda_s, device,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_sum += loss.item()
            lf_sum += lf
            ls_sum += ls
            n += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{total_sum / n:.4f}",
                L_f=f"{lf:.4f}",
                L_s=f"{ls:.4f}",
            )
        n = max(n, 1)
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Loss: {total_sum / n:.4f} | L_f: {lf_sum / n:.4f} | L_s: {ls_sum / n:.4f}"
        )

    # Export final probe (fixed z=0 for reproducible universal probe)
    G.eval()
    with torch.no_grad():
        z_final = torch.zeros(1, args.latent_dim, device=device)
        w_final = G(z_final)
        s_probe_final = G.probe(w_final, pool).squeeze(0).cpu()
    s_probe_final = s_probe_final / (s_probe_final.abs().max() + 1e-8) * 0.95
    torchaudio.save(str(probe_path), s_probe_final.unsqueeze(0), SAMPLE_RATE)
    print(f"Saved Universal Adversarial Probe: {probe_path}")

    torch.save(
        {"G": G.state_dict(), "pool_size": args.pool_size, "latent_dim": args.latent_dim},
        out_dir / "G.pt",
    )
    print("Done.")


if __name__ == "__main__":
    main()
