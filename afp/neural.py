"""Candidate B: neural embedding fingerprints (NAFP/NMFP family).

Our own implementation of the NAFP architecture (Chang et al., ICASSP 2021),
structured so the published NMFP checkpoints (Araz et al., ISMIR 2025,
https://zenodo.org/records/15719945, GPL/AGPL weights - internal evaluation
only) restore into it. Input features follow the NMFP inference settings:
8 kHz, 1 s segments every 0.5 s, 256 slaney-mel bands 160-4000 Hz from a
1024-point magnitude STFT (hop 256, centred frames), dB re. max, 80 dB range,
scaled to [-1, 1].

Search: exact inner-product nearest neighbours per query segment, then
sequence re-scoring: every (track, start) candidate is scored by the mean
similarity of all query segments at that alignment (NAFP matcher).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .index import Match


@dataclass(frozen=True)
class NeuralConfig:
    sr: int = 8000
    seg_s: float = 1.0
    hop_s: float = 0.5
    n_fft: int = 1024
    stft_hop: int = 256
    n_mels: int = 256
    f_min: float = 160.0
    f_max: float = 4000.0
    dyn_range: float = 80.0
    amin: float = 1e-5
    emb: int = 128
    top_k: int = 20          # neighbours per query segment

    @property
    def hop(self) -> int:     # used by the evaluator to convert offsets to seconds
        return int(self.hop_s * self.sr)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- features

_MEL_CACHE: dict = {}


def mel_matrix(cfg: NeuralConfig) -> np.ndarray:
    """Essentia MelBands (slaney, unit_tri) as a matrix; it is linear in its input."""
    key = (cfg.sr, cfg.n_fft, cfg.n_mels, cfg.f_min, cfg.f_max)
    if key not in _MEL_CACHE:
        import essentia.standard as es
        mb = es.MelBands(highFrequencyBound=cfg.f_max, inputSize=cfg.n_fft // 2 + 1, log=False,
                         lowFrequencyBound=cfg.f_min, normalize="unit_tri", numberBands=cfg.n_mels,
                         sampleRate=cfg.sr, type="magnitude", warpingFormula="slaneyMel", weighting="linear")
        eye = np.eye(cfg.n_fft // 2 + 1, dtype=np.float32)
        _MEL_CACHE[key] = np.stack([mb(row) for row in eye]).astype(np.float32)  # (bins, mels)
    return _MEL_CACHE[key]


def segment(x: np.ndarray, cfg: NeuralConfig) -> np.ndarray:
    L, H = int(cfg.seg_s * cfg.sr), int(cfg.hop_s * cfg.sr)
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    n = (len(x) - L) // H + 1
    idx = np.arange(L)[None, :] + H * np.arange(n)[:, None]
    return x[idx].astype(np.float32)


def melspec(segs: np.ndarray, cfg: NeuralConfig) -> np.ndarray:
    """(B, L) segments -> (B, n_mels, T, 1) scaled log-mel, each segment centred and zero padded."""
    # Essentia FrameGenerator(startFromZero=False, lastFrameToEndOfFile=True): frames centred on
    # 0, hop, 2*hop, ... until a centre passes the end; zero padding outside the segment.
    pad = cfg.n_fft // 2
    xp = np.pad(segs, ((0, 0), (pad, cfg.n_fft)))
    n_frames = 1 + int(np.ceil(segs.shape[1] / cfg.stft_hop))
    starts = cfg.stft_hop * np.arange(n_frames)
    frames = xp[:, starts[:, None] + np.arange(cfg.n_fft)[None, :]]           # (B, T, n_fft)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(cfg.n_fft) / cfg.n_fft)).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * win, axis=-1)).astype(np.float32)        # (B, T, bins)
    mel = mag @ mel_matrix(cfg)                                                # (B, T, mels)
    mel = np.maximum(mel, cfg.amin)
    db = 20 * np.log10(mel / mel.max(axis=(1, 2), keepdims=True))
    db = np.maximum(db, -cfg.dyn_range)
    scaled = 1 + db / (cfg.dyn_range / 2)
    return np.transpose(scaled, (0, 2, 1))[..., None].astype(np.float32)


# ---------------------------------------------------------------- model

def _tf():
    # The checkpoints were written by tf.keras 2; Keras 3 names Sequential sub-objects
    # differently, so restore with the legacy implementation.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf
    return tf


def build_model(cfg: NeuralConfig):
    tf = _tf()

    class ConvLayer(tf.keras.layers.Layer):
        def __init__(self, ch, strides):
            super().__init__()
            self.conv2d_1x3 = tf.keras.layers.Conv2D(ch, (1, 3), strides=strides[0], padding="SAME")
            self.conv2d_3x1 = tf.keras.layers.Conv2D(ch, (3, 1), strides=strides[1], padding="SAME")
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
            self.layer = tf.keras.Sequential([self.conv2d_1x3, tf.keras.layers.ELU(), self.BN_1x3,
                                              self.conv2d_3x1, tf.keras.layers.ELU(), self.BN_3x1])

        def call(self, x):
            return self.layer(x)

    class DivEncLayer(tf.keras.layers.Layer):
        def __init__(self, q, unit=(32, 1)):
            super().__init__()
            self.q = q
            self.BN = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(q)]
            self.split_fc_layers = [tf.keras.Sequential([tf.keras.layers.Dense(unit[0], activation="elu"),
                                                         self.BN[i], tf.keras.layers.Dense(unit[1])])
                                    for i in range(q)]

        def call(self, x):
            xs = tf.reshape(x, [tf.shape(x)[0], self.q, -1])
            return tf.concat([self.split_fc_layers[i](xs[:, i, :]) for i in range(self.q)], axis=1)

    class FingerPrinter(tf.keras.Model):
        def __init__(self):
            super().__init__()
            ch = [128, 128, 256, 256, 512, 512, 1024, 1024]
            st = [[(1, 2), (2, 1)]] * 4 + [[(1, 1), (2, 1)], [(1, 2), (2, 1)], [(1, 1), (2, 1)], [(1, 2), (2, 1)]]
            self.front_conv = tf.keras.Sequential(name="ConvLayers")
            for c, s in zip(ch, st):
                self.front_conv.add(ConvLayer(c, s))
            self.front_conv.add(tf.keras.layers.Flatten())
            self.div_enc = DivEncLayer(cfg.emb)

        def call(self, x):
            return tf.math.l2_normalize(self.div_enc(self.front_conv(x)), axis=1)

    return FingerPrinter()


class Embedder:
    def __init__(self, ckpt: str, cfg: NeuralConfig = NeuralConfig()):
        tf = _tf()
        self.cfg = cfg
        self.model = build_model(cfg)
        T = 1 + int(np.ceil(cfg.seg_s * cfg.sr / cfg.stft_hop))
        self.model(np.zeros((1, cfg.n_mels, T, 1), np.float32))
        status = tf.train.Checkpoint(model=self.model).restore(ckpt)
        status.expect_partial()
        status.assert_existing_objects_matched()
        self._fn = tf.function(lambda x: self.model(x, training=False),
                               input_signature=[tf.TensorSpec([None, cfg.n_mels, T, 1], tf.float32)])

    def __call__(self, x: np.ndarray, batch: int = 64) -> np.ndarray:
        segs = segment(x, self.cfg)
        out = [self._fn(melspec(segs[i:i + batch], self.cfg)).numpy() for i in range(0, len(segs), batch)]
        return np.concatenate(out).astype(np.float32) if out else np.empty((0, self.cfg.emb), np.float32)


# ---------------------------------------------------------------- index

class NeuralIndex:
    kind = "neural"

    def __init__(self, ckpt: str, cfg: NeuralConfig = NeuralConfig()):
        self.cfg = cfg
        self.ckpt = ckpt
        self.embedder = Embedder(ckpt, cfg)
        self.names: list[str] = []
        self._pending: list[np.ndarray] = []
        self.emb = np.empty((0, cfg.emb), np.float32)
        self.tracks = np.empty(0, np.int32)
        self.seg = np.empty(0, np.int32)
        self.starts = np.zeros(1, np.int64)   # first row of each track
        self._faiss = None
        self._mask_tracks: set[int] = set()

    def add(self, name: str, x: np.ndarray) -> int:
        e = self.embedder(x)
        self.names.append(name)
        self._pending.append(e)
        return len(e)

    def finalize(self) -> None:
        if not self._pending:
            return
        allv = [self.emb] + self._pending
        sizes = [len(p) for p in self._pending]
        base = len(self.names) - len(self._pending)
        self.emb = np.concatenate(allv)
        self.tracks = np.concatenate([self.tracks] + [np.full(n, base + i, np.int32) for i, n in enumerate(sizes)])
        self.seg = np.concatenate([self.seg] + [np.arange(n, dtype=np.int32) for n in sizes])
        self.starts = np.r_[self.starts, self.starts[-1] + np.cumsum(sizes)]
        self._pending = []
        self._faiss = None

    def __len__(self) -> int:
        return len(self.emb)

    def nbytes(self) -> int:
        return 4 * self.cfg.emb * len(self.emb)  # float32; PQ would make this 8-16x smaller

    def remove_tracks(self, names: set[str]) -> None:
        self._mask_tracks |= {i for i, n in enumerate(self.names) if n in names}

    def _index(self):
        if self._faiss is None:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.cfg.emb)
            self._faiss.add(self.emb)
        return self._faiss

    def query(self, x: np.ndarray, top_k: int = 5, tolerance: int = 1) -> list[Match]:
        q = self.embedder(x)
        if len(q) == 0 or len(self.emb) == 0:
            return []
        k = self.cfg.top_k + 64 * bool(self._mask_tracks)
        _, I = self._index().search(q, k)
        cand = {}
        for i in range(len(q)):
            for j in I[i]:
                if j < 0:
                    continue
                t = int(self.tracks[j])
                if t in self._mask_tracks:
                    continue
                cand[(t, int(self.seg[j]) - i)] = None
        best: dict[int, tuple[float, int]] = {}
        n_q = len(q)
        for (t, s) in cand:
            n_t = self.starts[t + 1] - self.starts[t]
            ii = np.arange(n_q)
            pos = s + ii
            ok = (pos >= 0) & (pos < n_t)
            if not ok.any():
                continue
            sims = np.einsum("ij,ij->i", q[ii[ok]], self.emb[self.starts[t] + pos[ok]])
            score = float(sims.sum() / n_q)   # missing segments count as 0 similarity
            if t not in best or score > best[t][0]:
                best[t] = (score, s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[: max(top_k, 2)]
        out = []
        for r, (t, (sc, s)) in enumerate(ranked[:top_k]):
            second = ranked[1][1][0] if r == 0 and len(ranked) > 1 else (ranked[0][1][0] if r > 0 else 0.0)
            out.append(Match(t, self.names[t], sc, s, second))
        return out

    def save(self, path: str) -> None:
        self.finalize()
        np.savez(path, emb=self.emb, tracks=self.tracks, seg=self.seg, starts=self.starts,
                 meta=np.frombuffer(json.dumps({"kind": self.kind, "names": self.names, "ckpt": self.ckpt,
                                                "cfg": self.cfg.to_dict()}).encode(), np.uint8))

    @classmethod
    def load(cls, path: str) -> "NeuralIndex":
        d = np.load(path)
        meta = json.loads(bytes(d["meta"]).decode())
        idx = cls(meta["ckpt"], NeuralConfig(**meta["cfg"]))
        idx.names = meta["names"]
        idx.emb, idx.tracks, idx.seg, idx.starts = d["emb"], d["tracks"], d["seg"], d["starts"]
        return idx
