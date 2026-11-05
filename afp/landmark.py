"""Shazam-style landmark fingerprints.

Pipeline: STFT -> local spectral peaks ("constellation map") -> pairs of nearby
peaks -> 32-bit hash (f_anchor, f_target, dt) stored with the anchor time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter, uniform_filter


@dataclass(frozen=True)
class LandmarkConfig:
    sr: int = 8000
    n_fft: int = 1024          # 128 ms window
    hop: int = 128             # 16 ms hop
    f_min_bin: int = 10        # ~78 Hz; phone mics are unreliable below this
    f_max_bin: int = 511       # ~3.99 kHz; keeps f in 9 bits
    peak_f_size: int = 21      # local-max neighbourhood (bins)
    peak_t_size: int = 11      # local-max neighbourhood (frames)
    peaks_per_sec: float = 30.0
    normalize: bool = True     # subtract local spectral mean before peak picking
    dt_min: int = 2            # target zone, frames
    dt_max: int = 63           # 6 bits -> up to ~1 s
    df_max: int = 127          # max |f_target - f_anchor| (bins)
    fanout: int = 5            # pairs per anchor

    def to_dict(self) -> dict:
        return asdict(self)


F_BITS = 9
DT_BITS = 6


def spectrogram(x: np.ndarray, cfg: LandmarkConfig) -> np.ndarray:
    """Log-magnitude spectrogram, shape (freq_bins, frames)."""
    if len(x) < cfg.n_fft:
        x = np.pad(x, (0, cfg.n_fft - len(x)))
    frames = sliding_window_view(x, cfg.n_fft)[:: cfg.hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(cfg.n_fft), axis=1)).T
    return np.log(spec + 1e-6).astype(np.float32)


def find_peaks(logspec: np.ndarray, cfg: LandmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_idx, freq_bin) of selected peaks, sorted by time."""
    s = logspec[: cfg.f_max_bin + 1]
    if cfg.normalize:
        # Local normalisation: peaks must stand out from their neighbourhood,
        # which helps under coloured noise and mic EQ.
        s = s - uniform_filter(s, size=(31, 63), mode="nearest")
    is_max = s == maximum_filter(s, size=(cfg.peak_f_size, cfg.peak_t_size), mode="nearest")
    is_max[: cfg.f_min_bin] = False
    f_idx, t_idx = np.nonzero(is_max)
    vals = s[f_idx, t_idx]
    if len(vals) == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)

    # Density control: keep the strongest peaks inside each 1-second block.
    frames_per_sec = cfg.sr / cfg.hop
    block = (t_idx / frames_per_sec).astype(np.int64)
    per_block = max(1, int(round(cfg.peaks_per_sec)))
    order = np.lexsort((-vals, block))
    block_sorted = block[order]
    starts = np.r_[0, np.flatnonzero(np.diff(block_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = order[rank < per_block]

    t, f = t_idx[keep], f_idx[keep]
    o = np.lexsort((f, t))
    return t[o].astype(np.int32), f[o].astype(np.int32)


def pair_peaks(t: np.ndarray, f: np.ndarray, cfg: LandmarkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair each anchor with up to `fanout` later peaks in its target zone."""
    n = len(t)
    if n < 2:
        e = np.empty(0, np.int32)
        return e, e, e, e
    window = min(n - 1, int(np.ceil(cfg.peaks_per_sec * (cfg.dt_max + 1) * cfg.hop / cfg.sr)) * 2 + 8)
    i = np.arange(n)[:, None]
    j = i + np.arange(1, window + 1)[None, :]
    valid = j < n
    j = np.where(valid, j, 0)
    dt = t[j] - t[i]
    df = f[j] - f[i]
    valid &= (dt >= cfg.dt_min) & (dt <= cfg.dt_max) & (np.abs(df) <= cfg.df_max)
    # Take the first `fanout` valid targets of each anchor (closest in time).
    valid &= np.cumsum(valid, axis=1) <= cfg.fanout
    ai, ki = np.nonzero(valid)
    tj = j[ai, ki]
    return t[ai], f[ai], f[tj], (t[tj] - t[ai])


def pack_hash(f1: np.ndarray, f2: np.ndarray, dt: np.ndarray) -> np.ndarray:
    return ((f1.astype(np.uint32) << (F_BITS + DT_BITS)) | (f2.astype(np.uint32) << DT_BITS) | dt.astype(np.uint32))


def unpack_hash(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = h.astype(np.uint32)
    return (h >> (F_BITS + DT_BITS)) & 0x1FF, (h >> DT_BITS) & 0x1FF, h & 0x3F


def fingerprint(x: np.ndarray, cfg: LandmarkConfig = LandmarkConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Audio -> (hashes uint32, anchor_frames uint32)."""
    t, f = find_peaks(spectrogram(x, cfg), cfg)
    ta, f1, f2, dt = pair_peaks(t, f, cfg)
    return pack_hash(f1, f2, dt), ta.astype(np.uint32)
