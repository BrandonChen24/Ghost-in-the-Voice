"""
GiV training constants and CLI arguments.
"""
from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# Constants (data and model)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
AUDIO_LENGTH = 48000  # Fixed-length sample count, ~3 s at 16 kHz
POOL_SIZE = 100
LATENT_DIM = 64

LIBRISPEECH_URL = "train-clean-100"
HUBERT_ID = "facebook/hubert-base-ls960"
CONVTASNET_ID = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"


def parse_args() -> argparse.Namespace:
    """Parse training script CLI arguments."""
    p = argparse.ArgumentParser(description="GiV Adversarial Blending Training")
    # Data
    p.add_argument("--librispeech_root", type=str, default="./data", help="LibriSpeech root directory")
    p.add_argument("--librispeech_url", type=str, default=LIBRISPEECH_URL, help="Subset, e.g. train-clean-100")
    p.add_argument("--audio_length", type=int, default=AUDIO_LENGTH, help="Samples per audio segment")
    p.add_argument("--pool_size", type=int, default=POOL_SIZE, help="Candidate pool size")
    # Model
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM, help="Noise z dimension")
    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_f", type=float, default=1.0, help="Feature disruption loss L_f weight")
    p.add_argument("--lambda_s", type=float, default=1.0, help="Separation suppression loss L_s weight")
    # Output and device
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--probe_path", type=str, default="v_probe.wav")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device. Default: cuda if GPU available else cpu; override with cuda or cpu",
    )
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return p.parse_args()


def get_device(args: argparse.Namespace) -> str:
    """Prefer GPU: use cuda if available else cpu; if --device is set, use that."""
    if args.device:
        return args.device
    return "cuda" if __import__("torch").cuda.is_available() else "cpu"
