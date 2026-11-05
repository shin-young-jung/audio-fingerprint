"""Synthetic degradations that approximate a phone recording music from a speaker.

Order (as in the physical chain): speaker band-limit -> room reverb -> background
noise -> phone mic band-limit -> gain -> optional lossy codec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .audio import encode_decode


@dataclass(frozen=True)
class Degradation:
    name: str
    snr_db: float | None = None     # None = no added noise
    rt60: float | None = None       # reverb time (s); None = dry
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    codec: str | None = None        # "aac" | "opus" | "mp3"
    bitrate: str = "32k"


PRESETS: dict[str, Degradation] = {
    "clean": Degradation("clean"),
    "noise_10db": Degradation("noise_10db", snr_db=10),
    "noise_0db": Degradation("noise_0db", snr_db=0),
    "reverb": Degradation("reverb", rt60=0.6),
    "phone_quiet_room": Degradation("phone_quiet_room", snr_db=15, rt60=0.4, highpass_hz=200, lowpass_hz=3500),
    "phone_cafe": Degradation("phone_cafe", snr_db=5, rt60=0.6, highpass_hz=200, lowpass_hz=3500, codec="aac"),
    "phone_loud_bar": Degradation("phone_loud_bar", snr_db=0, rt60=0.9, highpass_hz=300, lowpass_hz=3000, codec="aac"),
}


def synthetic_rir(sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Exponentially decaying noise RIR with a direct-path impulse."""
    n = int(sr * min(rt60 * 1.2, 2.0))
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[: int(0.003 * sr)] = 0.0       # gap before first reflections
    rir[0] = 1.0 / 0.3                  # direct path dominates (~10 dB DRR)
    return (rir / np.sqrt(np.sum(rir**2))).astype(np.float32)


def band_limit(x: np.ndarray, sr: int, hp: float | None, lp: float | None) -> np.ndarray:
    if hp:
        x = sosfilt(butter(4, hp, "highpass", fs=sr, output="sos"), x)
    if lp and lp < sr / 2:
        x = sosfilt(butter(6, lp, "lowpass", fs=sr, output="sos"), x)
    return x


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    px = np.mean(x**2) + 1e-12
    pn = np.mean(noise**2) + 1e-12
    return x + noise * np.sqrt(px / (pn * 10 ** (snr_db / 10)))


def apply(x: np.ndarray, sr: int, deg: Degradation, noise_bank: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float64)
    if deg.rt60:
        y = fftconvolve(y, synthetic_rir(sr, deg.rt60, rng))[: len(x)]
    if deg.snr_db is not None:
        if noise_bank:
            src = noise_bank[rng.integers(len(noise_bank))]
            if len(src) < len(y):
                src = np.tile(src, int(np.ceil(len(y) / len(src))))
            s = rng.integers(0, len(src) - len(y) + 1)
            noise = src[s: s + len(y)].astype(np.float64)
        else:
            noise = rng.standard_normal(len(y))
        y = mix_at_snr(y, noise, deg.snr_db)
    y = band_limit(y, sr, deg.highpass_hz, deg.lowpass_hz)
    y = y * 10 ** (rng.uniform(-12, 0) / 20)  # random playback level
    y = y / max(1.0, np.max(np.abs(y)))
    y = y.astype(np.float32)
    if deg.codec:
        y = encode_decode(y, sr, deg.codec, deg.bitrate)
    return y
