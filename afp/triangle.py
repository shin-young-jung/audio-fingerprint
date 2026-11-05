"""Candidate A: triangle (3-landmark) hashing with query-side expansion.

Re-implementation of the POLARIS method (J. Li, arXiv:2609.14820, MIT licence,
https://github.com/JihengLi/POLARIS) inside this project's evaluation harness,
adapted to 8 kHz input. Same time/frequency resolution as the paper
(64 ms window, 32 ms hop, 15.6 Hz bins); the frequency band is moved down
because 8 kHz audio has nothing above 4 kHz.

Reference side : saliency landmarks -> Delaunay faces -> 40-bit triangle hash.
Query side     : 4 shifted STFTs, +-1 multiprobe on every hash coordinate,
                 optional two-hop triangles (adaptive: only when unsure).
Matching       : IDF-weighted time-offset votes with neighbour smoothing.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from math import log

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter, gaussian_filter1d, generate_binary_structure, label, maximum_filter
from scipy.spatial import Delaunay, QhullError

from .index import Match


@dataclass(frozen=True)
class TriangleConfig:
    sr: int = 8000
    n_fft: int = 512                 # 64 ms
    hop: int = 256                   # 32 ms
    query_shifts: int = 4
    # landmarks (paper values unless noted)
    min_bin: int = 25                # 390 Hz  (paper: 781 Hz at 40 kHz input)
    max_bin: int = 250               # 3.9 kHz (paper: 5.5 kHz; 8 kHz input tops out at 4 kHz)
    magnitude_threshold: float = 10.0
    saliency_threshold: float = 5.0
    max_f_radius: int = 2
    max_t_radius: int = 3
    fine_sigma: tuple[float, float] = (1.0, 1.5)
    background_sigma: tuple[float, float] = (8.0, 16.0)
    norm_sigma: tuple[float, float] = (6.0, 12.0)
    contrast_floor: float = 0.75
    contrast_weight: float = 3.0
    magnitude_weight: float = 0.015
    landmarks_per_sec: float = 22.0
    min_landmarks_per_sec: float = 6.0
    max_landmarks_per_sec: float = 44.0
    density_window_s: float = 2.0
    density_background_s: float = 8.0
    density_exponent: float = 0.75
    # triangles
    f_scale: float = 10.0
    t_scale: float = 14.0
    min_t_span: int = 1
    max_t_span: int = 63
    min_area: float = 0.025
    max_edge: float = 6.0
    max_circumradius: float = 4.0
    anchor_f_q: float = 2.0
    delta_f_q: float = 3.0
    t_q: int = 2
    # query
    hops: int = 2
    max_neighbors: int = 12
    max_extra_per_anchor: int = 24
    multiprobe: int = 1
    # matcher
    offset_radius: int = 1
    offset_neighbor_weight: float = 0.5
    adaptive_min_hashes: int = 8
    adaptive_min_margin: float = 0.40

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- landmarks

def _db_spectrogram(x: np.ndarray, cfg: TriangleConfig) -> np.ndarray:
    pad = cfg.n_fft // 2
    x = np.pad(x.astype(np.float64), (pad, pad), mode="reflect" if len(x) > pad else "constant")
    frames = sliding_window_view(x, cfg.n_fft)[:: cfg.hop]
    mag = np.abs(np.fft.rfft(frames * np.hanning(cfg.n_fft + 1)[:-1], axis=1)).T
    ref = max(mag.max(), 1e-10)
    db = 20 * np.log10(np.maximum(mag, 1e-10) / ref)
    return (np.maximum(db, -80.0) + 80.0).astype(np.float32)


def _local_maxima(v: np.ndarray, fr: int, tr: int) -> np.ndarray:
    m = maximum_filter(v, size=(2 * fr + 1, 2 * tr + 1), mode="nearest")
    labels, n = label(v == m, structure=generate_binary_structure(2, 2))
    fl, fv = labels.ravel(), v.ravel()
    cand = np.flatnonzero(fl)
    if cand.size == 0:
        return np.empty((0, 2), np.int64)
    best_val = np.full(n + 1, -np.inf)
    np.maximum.at(best_val, fl[cand], fv[cand])
    best = cand[fv[cand] == best_val[fl[cand]]]
    _, first = np.unique(fl[best], return_index=True)
    f, t = np.unravel_index(best[first], v.shape)
    return np.column_stack((f, t))


def landmarks(x: np.ndarray, cfg: TriangleConfig) -> list[tuple[int, int, float]]:
    """Return (freq_bin, time_bin, saliency) chosen by the paper's rate controller."""
    spec = _db_spectrogram(x, cfg)
    lo, hi = cfg.min_bin, min(spec.shape[0], cfg.max_bin)
    band = spec[lo:hi]
    if band.size == 0 or band.shape[1] == 0:
        return []
    sm = gaussian_filter(band, sigma=cfg.fine_sigma, mode="nearest")
    contrast = sm - gaussian_filter(sm, sigma=cfg.background_sigma, mode="nearest")
    energy = gaussian_filter(contrast * contrast, sigma=cfg.norm_sigma, mode="nearest")
    sal = cfg.contrast_weight * contrast / np.sqrt(energy + cfg.contrast_floor**2)
    sal += cfg.magnitude_weight * np.maximum(sm - cfg.magnitude_threshold, 0.0)
    sal = np.clip(sal, -12.0, 12.0)

    coords = _local_maxima(sal, cfg.max_f_radius, cfg.max_t_radius)
    if coords.size == 0:
        return []
    scores = sal[coords[:, 0], coords[:, 1]].astype(np.float64)
    keep = scores > cfg.saliency_threshold
    coords, scores = coords[keep], scores[keep]
    order = np.lexsort((coords[:, 0], coords[:, 1], -scores))

    # Adaptive density: more landmarks where the music is locally more salient.
    pos = np.maximum(sal, 0.0)
    top = max(1, pos.shape[0] // 20)
    temporal = np.partition(pos, pos.shape[0] - top, axis=0)[-top:].mean(axis=0)
    fps = cfg.sr / cfg.hop
    fg_sigma = max(1.0, cfg.density_window_s * fps / 4)
    bg_sigma = max(fg_sigma, cfg.density_background_s * fps / 4)
    rel = (gaussian_filter1d(temporal, fg_sigma, mode="nearest") + 0.25) / (
        gaussian_filter1d(temporal, bg_sigma, mode="nearest") + 0.25)
    rates = np.clip(cfg.landmarks_per_sec * rel**cfg.density_exponent,
                    cfg.min_landmarks_per_sec, cfg.max_landmarks_per_sec)
    window = cfg.density_window_s * fps

    selected: list[tuple[int, int, float]] = []
    times: list[int] = []
    for i in order:
        f, t, s = int(coords[i, 0]), int(coords[i, 1]), float(scores[i])
        cap = max(1, round(rates[t] * cfg.density_window_s))
        b = bisect_left(times, t - window)
        e = bisect_right(times, t)
        starts = times[b:e] + [t]
        if any(bisect_right(times, st + window) - bisect_left(times, st) >= cap for st in starts):
            continue
        selected.append((f + lo, t, s))
        insort(times, t)
    return selected


# ---------------------------------------------------------------- triangles

def _pack(desc: tuple[int, int, int, int, int]) -> int:
    a, *rest = desc
    if not 0 <= a <= 255 or any(not -128 <= v <= 127 for v in rest):
        return -1
    k = a
    for v in rest:
        k = (k << 8) | (v + 128)
    return k


class _Geometry:
    def __init__(self, pts: list[tuple[int, int, float]], cfg: TriangleConfig):
        uniq: dict[tuple[int, int], tuple[int, int, float]] = {}
        for p in pts:
            q = uniq.get((p[0], p[1]))
            if q is None or p[2] > q[2]:
                uniq[(p[0], p[1])] = p
        self.pts = sorted(uniq.values(), key=lambda p: (p[1], p[0], -p[2]))
        self.cfg = cfg
        self.xy = np.array([(p[0] / cfg.f_scale, p[1] / cfg.t_scale) for p in self.pts], np.float64).reshape(-1, 2)
        self.faces: set[tuple[int, int, int]] = set()
        self.adj: list[set[int]] = []
        self._rank_cache: dict = {}
        if len(self.pts) < 3:
            return
        try:
            tri = Delaunay(self.xy)
        except QhullError:
            return
        self.faces = {tuple(sorted(int(i) for i in s)) for s in tri.simplices}
        self.adj = [set() for _ in self.pts]
        for a, b, c in self.faces:
            self.adj[a].update((b, c)); self.adj[b].update((a, c)); self.adj[c].update((a, b))

    def _rank(self, idx):
        if idx in self._rank_cache:
            return self._rank_cache[idx]
        cfg, P = self.cfg, self.pts
        r = None
        span = P[idx[2]][1] - P[idx[0]][1]
        if cfg.min_t_span <= span <= cfg.max_t_span:
            c = self.xy[list(idx)]
            u, v = c[1] - c[0], c[2] - c[0]
            area = abs(u[0] * v[1] - u[1] * v[0]) / 2
            if area >= cfg.min_area:
                e = (np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1]), np.linalg.norm(c[0] - c[2]))
                if max(e) <= cfg.max_edge:
                    R = e[0] * e[1] * e[2] / (4 * area)
                    if R <= cfg.max_circumradius:
                        quality = 4 * np.sqrt(3) * area / sum(x * x for x in e)
                        r = (-quality, -min(P[i][2] for i in idx), R, idx)
        self._rank_cache[idx] = r
        return r

    def _hash(self, tris, probe: int) -> set[tuple[int, int]]:
        cfg, P, out = self.cfg, self.pts, set()
        for i0, i1, i2 in tris:
            a, b, c = P[i0], P[i1], P[i2]
            d = [round(a[0] / cfg.anchor_f_q), round((b[0] - a[0]) / cfg.delta_f_q), round((b[1] - a[1]) / cfg.t_q),
                 round((c[0] - a[0]) / cfg.delta_f_q), round((c[1] - a[1]) / cfg.t_q)]
            probes = {tuple(d)}
            for dim in range(5):
                for dist in range(1, probe + 1):
                    for sgn in (-1, 1):
                        s = d.copy(); s[dim] += sgn * dist; probes.add(tuple(s))
            for p in probes:
                k = _pack(p)
                if k >= 0:
                    out.add((k, a[1]))
        return out

    def face_hashes(self, probe: int) -> set[tuple[int, int]]:
        return self._hash([f for f in self.faces if self._rank(f) is not None], probe)

    def two_hop_hashes(self, probe: int) -> set[tuple[int, int]]:
        cfg, P = self.cfg, self.pts
        chosen = []
        for ai in range(len(P)):
            if not self.adj:
                break
            seen, frontier = {ai}, {ai}
            for _ in range(cfg.hops):
                frontier = {n for i in frontier for n in self.adj[i] if n not in seen}
                if not frontier:
                    break
                seen |= frontier
            nb = [i for i in seen if i > ai and P[i][1] - P[ai][1] <= cfg.max_t_span]
            nb.sort(key=lambda i: (float(np.linalg.norm(self.xy[i] - self.xy[ai])), -P[i][2], P[i][1], P[i][0]))
            ranked = []
            for x, y in combinations(nb[: cfg.max_neighbors], 2):
                idx = tuple(sorted((ai, x, y)))
                if idx in self.faces:
                    continue
                r = self._rank(idx)
                if r is not None:
                    ranked.append((r, idx))
            chosen += [idx for _, idx in sorted(ranked)[: cfg.max_extra_per_anchor]]
        return self._hash(chosen, probe)


# ---------------------------------------------------------------- index

class TriangleIndex:
    kind = "triangle"

    def __init__(self, cfg: TriangleConfig = TriangleConfig(), mode: str = "a"):
        self.cfg = cfg
        self.mode = mode  # "o" original faces only, "a" adaptive two-hop, "f" always two-hop
        self.names: list[str] = []
        self._pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.keys = np.empty(0, np.uint64)
        self.tracks = np.empty(0, np.uint32)
        self.times = np.empty(0, np.uint32)
        self._df: np.ndarray | None = None

    def add(self, name: str, x: np.ndarray) -> int:
        g = _Geometry(landmarks(x, self.cfg), self.cfg)
        hs = g.face_hashes(probe=0)
        k = np.fromiter((h for h, _ in hs), np.uint64, len(hs))
        t = np.fromiter((t for _, t in hs), np.uint32, len(hs))
        tid = len(self.names)
        self.names.append(name)
        self._pending.append((k, np.full(len(k), tid, np.uint32), t))
        return len(k)

    def finalize(self) -> None:
        if not self._pending:
            return
        parts = [(self.keys, self.tracks, self.times)] + self._pending
        k = np.concatenate([p[0] for p in parts])
        tr = np.concatenate([p[1] for p in parts])
        tm = np.concatenate([p[2] for p in parts])
        o = np.lexsort((tm, tr, k))
        self.keys, self.tracks, self.times = k[o], tr[o], tm[o]
        self._pending = []
        self._df = None

    def __len__(self) -> int:
        return len(self.keys)

    def nbytes(self) -> int:
        return 13 * len(self.keys)  # 5 B key + 4 B track + 4 B time (prototype layout)

    def remove_tracks(self, names: set[str]) -> None:
        self.finalize()
        ids = np.array([i for i, n in enumerate(self.names) if n in names], np.uint32)
        keep = ~np.isin(self.tracks, ids)
        self.keys, self.tracks, self.times = self.keys[keep], self.tracks[keep], self.times[keep]
        self._df = None

    def _doc_freq(self) -> np.ndarray:
        """Number of distinct tracks per entry's key (for IDF), aligned with self.keys."""
        if self._df is None:
            if len(self.keys) == 0:
                self._df = np.empty(0, np.int64)
            else:
                new_key = np.r_[True, self.keys[1:] != self.keys[:-1]]
                new_pair = new_key | np.r_[True, self.tracks[1:] != self.tracks[:-1]]
                grp = np.cumsum(new_key) - 1
                per_key = np.bincount(grp, weights=new_pair).astype(np.int64)
                self._df = per_key[grp]
        return self._df

    def _vote(self, hashes: set[tuple[int, int]], votes: dict) -> None:
        if not hashes:
            return
        qk = np.fromiter((h for h, _ in hashes), np.uint64, len(hashes))
        qt = np.fromiter((t for _, t in hashes), np.int64, len(hashes))
        lo = np.searchsorted(self.keys, qk, "left")
        hi = np.searchsorted(self.keys, qk, "right")
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            return
        idx = np.repeat(lo, cnt) + (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        n_tracks = max(len(self.names), 1)
        df = self._doc_freq()[idx]
        w = 1.0 + np.log((n_tracks + 1.0) / (df + 1.0))
        trk = self.tracks[idx].astype(np.int64)
        off = self.times[idx].astype(np.int64) - np.repeat(qt, cnt)
        key = trk * (1 << 26) + (off + (1 << 24))
        uk, inv = np.unique(key, return_inverse=True)
        c = np.bincount(inv).astype(np.float64)
        s = np.bincount(inv, weights=w)
        for k_, c_, s_ in zip(uk.tolist(), c.tolist(), s.tolist()):
            cell = votes.get(k_)
            if cell is None:
                votes[k_] = [c_, s_]
            else:
                cell[0] += c_; cell[1] += s_

    def _rank(self, votes: dict, top_k: int) -> list[tuple[int, int, int, float]]:
        if not votes:
            return []
        keys = np.fromiter(votes.keys(), np.int64, len(votes))
        vals = np.array(list(votes.values()), np.float64)
        o = np.argsort(keys)
        keys, cnt, sc = keys[o], vals[o, 0].copy(), vals[o, 1].copy()
        tot_c, tot_s = cnt.copy(), sc.copy()
        for d in range(1, self.cfg.offset_radius + 1):
            wgt = self.cfg.offset_neighbor_weight**d
            for sgn in (-1, 1):
                p = np.minimum(np.searchsorted(keys, keys + sgn * d), len(keys) - 1)
                hit = keys[p] == keys + sgn * d
                tot_c += np.where(hit, cnt[p], 0)
                tot_s += np.where(hit, wgt * sc[p], 0)
        trk = keys >> 26
        off = (keys & ((1 << 26) - 1)) - (1 << 24)
        oo = np.lexsort((-tot_c, -tot_s, trk))
        first = np.r_[True, trk[oo][1:] != trk[oo][:-1]]
        best = oo[first]
        best = best[np.lexsort((-tot_c[best], -tot_s[best]))][:top_k]
        return [(int(trk[b]), int(off[b]), int(tot_c[b]), float(tot_s[b])) for b in best]

    def query(self, x: np.ndarray, top_k: int = 5, tolerance: int = 1) -> list[Match]:
        cfg = self.cfg
        geoms = []
        step = cfg.hop // cfg.query_shifts
        for s in range(cfg.query_shifts):
            geoms.append(_Geometry(landmarks(x[s * step:], cfg), cfg))
        original = set().union(*(g.face_hashes(cfg.multiprobe) for g in geoms))
        votes: dict = {}
        if self.mode == "f":
            original |= set().union(*(g.two_hop_hashes(cfg.multiprobe) for g in geoms))
        self._vote(original, votes)
        ranked = self._rank(votes, max(top_k, 2))
        if self.mode == "a" and not self._confident(ranked):
            extra = set().union(*(g.two_hop_hashes(cfg.multiprobe) for g in geoms)) - original
            self._vote(extra, votes)
            ranked = self._rank(votes, max(top_k, 2))
        out = []
        for r, (t, off, c, s) in enumerate(ranked[:top_k]):
            second = ranked[1][2] if r == 0 and len(ranked) > 1 else (ranked[0][2] if r > 0 else 0)
            m = Match(t, self.names[t], c, off, second)
            m.weighted = s  # type: ignore[attr-defined]
            out.append(m)
        return out

    def _confident(self, ranked) -> bool:
        if not ranked:
            return False
        first = ranked[0][3]
        second = ranked[1][3] if len(ranked) > 1 else 0.0
        margin = (first - second) / first if first > 0 else 0.0
        return ranked[0][2] >= self.cfg.adaptive_min_hashes and margin >= self.cfg.adaptive_min_margin

    def save(self, path: str) -> None:
        self.finalize()
        cfg = self.cfg.to_dict()
        np.savez(path, keys=self.keys, tracks=self.tracks, times=self.times,
                 meta=np.frombuffer(json.dumps({"kind": self.kind, "names": self.names, "cfg": cfg,
                                                "mode": self.mode}).encode(), np.uint8))

    @classmethod
    def load(cls, path: str) -> "TriangleIndex":
        d = np.load(path)
        meta = json.loads(bytes(d["meta"]).decode())
        cfg = meta["cfg"]
        for k in ("fine_sigma", "background_sigma", "norm_sigma"):
            cfg[k] = tuple(cfg[k])
        idx = cls(TriangleConfig(**cfg), meta.get("mode", "a"))
        idx.names = meta["names"]
        idx.keys, idx.tracks, idx.times = d["keys"], d["tracks"], d["times"]
        return idx
