"""Audio loading. Uses ffmpeg so any common format (wav, mp3, ogg, m4a) works."""

from __future__ import annotations

import subprocess

import numpy as np

DEFAULT_SR = 8000


def load_audio(path: str, sr: int = DEFAULT_SR, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode a file to mono float32 at `sr` Hz."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if offset > 0:
        cmd += ["-ss", f"{offset:.3f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def encode_decode(x: np.ndarray, sr: int, codec: str = "aac", bitrate: str = "32k") -> np.ndarray:
    """Round-trip audio through a lossy codec with ffmpeg (simulates app/network compression)."""
    fmt = {"aac": "adts", "opus": "ogg", "mp3": "mp3"}[codec]
    lib = {"aac": "aac", "opus": "libopus", "mp3": "libmp3lame"}[codec]
    out_sr = 48000 if codec == "opus" else sr
    enc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-ar", str(out_sr), "-c:a", lib, "-b:a", bitrate, "-f", fmt, "-"],
        input=x.astype(np.float32).tobytes(), check=True, capture_output=True,
    ).stdout
    dec = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "-", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        input=enc, check=True, capture_output=True,
    ).stdout
    y = np.frombuffer(dec, dtype=np.float32).copy()
    # Codecs add padding; keep the original length.
    if len(y) >= len(x):
        return y[: len(x)]
    return np.pad(y, (0, len(x) - len(y)))
