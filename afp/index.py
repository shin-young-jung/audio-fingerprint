"""In-memory inverted index: hash -> [(track_id, anchor_frame)], plus offset-vote matching."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .landmark import LandmarkConfig, fingerprint


@dataclass
class Match:
    track_id: int
    name: str
    score: int          # number of hashes agreeing on the same time offset
    offset_frames: int  # db_time - query_time
    second_score: int   # best score of any other track

    @property
    def margin(self) -> float:
        return (self.score - self.second_score) / max(self.score, 1)


class LandmarkIndex:
    def __init__(self, cfg: LandmarkConfig = LandmarkConfig(), max_postings: int = 0):
        self.cfg = cfg
        self.max_postings = max_postings  # 0 = no cap; >0 drops overly common hashes at query time
        self.names: list[str] = []
        self._pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.hashes = np.empty(0, np.uint32)
        self.tracks = np.empty(0, np.uint32)
        self.times = np.empty(0, np.uint32)

    # ---- building -------------------------------------------------------
    def add(self, name: str, x: np.ndarray) -> int:
        h, t = fingerprint(x, self.cfg)
        tid = len(self.names)
        self.names.append(name)
        self._pending.append((h, np.full(len(h), tid, np.uint32), t))
        return len(h)

    def finalize(self) -> None:
        if not self._pending:
            return
        parts = [(self.hashes, self.tracks, self.times)] + self._pending
        h = np.concatenate([p[0] for p in parts])
        tr = np.concatenate([p[1] for p in parts])
        tm = np.concatenate([p[2] for p in parts])
        o = np.argsort(h, kind="stable")
        self.hashes, self.tracks, self.times = h[o], tr[o], tm[o]
        self._pending = []

    def add_synthetic_distractors(self, n_tracks: int, sec_per_track: float = 210.0, seed: int = 0) -> None:
        """Append fake tracks whose hashes follow this index's real hash distribution.

        Stand-in for a large catalogue when real audio is unavailable. Random
        times remove song structure, so this models chance collisions only and
        is optimistic about false matches (no covers, remasters, shared samples).
        """
        self.finalize()
        rng = np.random.default_rng(seed)
        fps = self.cfg.sr / self.cfg.hop
        real_frames = sum(int(self.times[self.tracks == i].max()) + 1 for i in range(len(self.names)) if (self.tracks == i).any())
        rate = len(self.hashes) / real_frames  # hashes per frame
        n_frames = int(sec_per_track * fps)
        per_track = int(rate * n_frames)
        base = len(self.names)
        chunk = max(1, 20_000_000 // max(per_track, 1))
        for c0 in range(0, n_tracks, chunk):
            k = min(chunk, n_tracks - c0)
            pick = rng.integers(0, len(self.hashes), k * per_track)
            h = self.hashes[pick]
            tr = (base + c0 + np.repeat(np.arange(k), per_track)).astype(np.uint32)
            tm = rng.integers(0, n_frames, k * per_track).astype(np.uint32)
            self._pending.append((h, tr, tm))
        self.names.extend(f"__distractor_{i}" for i in range(n_tracks))
        self.finalize()

    def remove_tracks(self, names: set[str]) -> None:
        """Drop all entries of the given tracks (names stay so ids remain stable)."""
        self.finalize()
        ids = np.array([i for i, n in enumerate(self.names) if n in names], np.uint32)
        keep = ~np.isin(self.tracks, ids)
        self.hashes, self.tracks, self.times = self.hashes[keep], self.tracks[keep], self.times[keep]

    def __len__(self) -> int:
        return len(self.hashes)

    def nbytes(self) -> int:
        """Logical index size: 4 B hash + 4 B track + 4 B time per entry."""
        return 12 * len(self.hashes)

    # ---- querying -------------------------------------------------------
    def query(self, x: np.ndarray, top_k: int = 5, tolerance: int = 1) -> list[Match]:
        qh, qt = fingerprint(x, self.cfg)
        return self.query_hashes(qh, qt, top_k=top_k, tolerance=tolerance)

    def query_hashes(self, qh: np.ndarray, qt: np.ndarray, top_k: int = 5, tolerance: int = 1) -> list[Match]:
        if len(qh) == 0 or len(self.hashes) == 0:
            return []
        lo = np.searchsorted(self.hashes, qh, "left")
        hi = np.searchsorted(self.hashes, qh, "right")
        cnt = hi - lo
        if self.max_postings > 0:
            cnt = np.where(cnt > self.max_postings, 0, cnt)
        total = int(cnt.sum())
        if total == 0:
            return []
        starts = np.repeat(lo, cnt)
        within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        idx = starts + within
        trk = self.tracks[idx].astype(np.int64)
        off = self.times[idx].astype(np.int64) - np.repeat(qt.astype(np.int64), cnt)

        # Vote on (track, offset). Neighbouring offsets (+-tolerance) are merged
        # because frame quantisation can shift a hash by one frame.
        bias = 1 << 24
        key = trk * (1 << 26) + (off + bias)
        uk, uc = np.unique(key, return_counts=True)
        score = uc.copy()
        for d in range(1, tolerance + 1):
            for sgn in (-1, 1):
                pos = np.searchsorted(uk, uk + sgn * d)
                pos_c = np.minimum(pos, len(uk) - 1)
                hit = uk[pos_c] == uk + sgn * d
                score = score + np.where(hit, uc[pos_c], 0)
        utrk = uk >> 26
        uoff = (uk & ((1 << 26) - 1)) - bias

        # Best offset per track.
        o = np.lexsort((-score, utrk))
        first = np.r_[True, utrk[o][1:] != utrk[o][:-1]]
        best = o[first]
        best = best[np.argsort(-score[best], kind="stable")][: max(top_k, 2)]
        scores = score[best]
        out = []
        for rank, b in enumerate(best[:top_k]):
            second = int(scores[1]) if rank == 0 and len(scores) > 1 else (int(scores[0]) if rank > 0 else 0)
            out.append(Match(int(utrk[b]), self.names[int(utrk[b])], int(score[b]), int(uoff[b]), second))
        return out

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        self.finalize()
        np.savez(path, hashes=self.hashes, tracks=self.tracks, times=self.times,
                 meta=np.frombuffer(json.dumps({"names": self.names, "cfg": self.cfg.to_dict(),
                                                "max_postings": self.max_postings}).encode(), np.uint8))

    @classmethod
    def load(cls, path: str) -> "LandmarkIndex":
        d = np.load(path)
        meta = json.loads(bytes(d["meta"]).decode())
        idx = cls(LandmarkConfig(**meta["cfg"]), meta.get("max_postings", 0))
        idx.names = meta["names"]
        idx.hashes, idx.tracks, idx.times = d["hashes"], d["tracks"], d["times"]
        return idx
