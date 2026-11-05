"""Evaluation CLI.

  python -m afp.evaluate build     --out idx.npz [--method landmark|triangle] REF_FILES...
  python -m afp.evaluate synth     --index idx.npz --refs REF_FILES... [--negatives FILES...] [--noise FILES...]
  python -m afp.evaluate recording --index idx.npz --recordings REC.wav... --order order.csv --songs-dir DIR

Metrics
  top1        : best candidate is the right track (no threshold)
  accept@T    : right track AND score >= T (what a user would see as a hit)
  wrong@T     : wrong track with score >= T (in-DB queries)
  fp@T        : any track accepted for a query whose song is NOT in the DB
                (synth: --negatives; recording: --as-negatives, reported as wrong@T)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from .audio import load_audio
from .degrade import PRESETS, apply
from .index import LandmarkIndex
from .landmark import LandmarkConfig
from .triangle import TriangleConfig, TriangleIndex

THRESHOLDS = (5, 8, 12, 20)


def _name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_index(path: str):
    meta = json.loads(bytes(np.load(path)["meta"]).decode())
    return TriangleIndex.load(path) if meta.get("kind") == "triangle" else LandmarkIndex.load(path)


def build(args) -> None:
    if args.method == "triangle":
        cfg = TriangleConfig()
        idx = TriangleIndex(cfg)
    else:
        cfg = LandmarkConfig(peaks_per_sec=args.density, fanout=args.fanout)
        idx = LandmarkIndex(cfg)
    total_sec = 0.0
    t0 = time.perf_counter()
    for p in args.refs:
        x = load_audio(p, cfg.sr)
        total_sec += len(x) / cfg.sr
        idx.add(_name(p), x)
    idx.finalize()
    idx.save(args.out)
    hours = total_sec / 3600
    print(json.dumps({
        "tracks": len(idx.names), "audio_hours": round(hours, 3), "entries": len(idx),
        "hashes_per_sec": round(len(idx) / total_sec, 1),
        "bytes_per_hour": int(idx.nbytes() / hours), "build_sec": round(time.perf_counter() - t0, 1),
    }, indent=2))


def _score(results: list[dict], thresholds=THRESHOLDS) -> dict:
    n = len(results)
    out = {"n": n, "top1": 100 * np.mean([r["correct"] for r in results]) if n else 0.0}
    for t in thresholds:
        out[f"accept@{t}"] = 100 * np.mean([r["correct"] and r["score"] >= t for r in results]) if n else 0.0
        out[f"wrong@{t}"] = 100 * np.mean([(not r["correct"]) and r["score"] >= t for r in results]) if n else 0.0
    out["median_ms"] = float(np.median([r["ms"] for r in results])) if n else 0.0
    return out


def _run_query(idx: LandmarkIndex, q: np.ndarray, truth: str | None) -> dict:
    t0 = time.perf_counter()
    m = idx.query(q, top_k=2)
    ms = 1000 * (time.perf_counter() - t0)
    if not m:
        return {"correct": False, "score": 0, "pred": None, "ms": ms, "offset": None}
    return {"correct": truth is not None and m[0].name == truth, "score": m[0].score,
            "pred": m[0].name, "ms": ms, "offset": m[0].offset_frames}


def _load(args):
    idx = load_index(args.index)
    if getattr(args, "mode", None) and isinstance(idx, TriangleIndex):
        idx.mode = args.mode
    if getattr(args, "distractors", 0):
        t0 = time.perf_counter()
        idx.add_synthetic_distractors(args.distractors)
        print(f"# added {args.distractors} synthetic distractor tracks -> {len(idx):,} entries "
              f"({idx.nbytes() / 1e9:.2f} GB) in {time.perf_counter() - t0:.0f}s", flush=True)
    return idx


def synth(args) -> None:
    idx = _load(args)
    sr = idx.cfg.sr
    rng = np.random.default_rng(args.seed)
    noise_bank = [load_audio(p, sr) for p in (args.noise or [])]
    refs = [(p, load_audio(p, sr)) for p in args.refs]
    negs = [(p, load_audio(p, sr)) for p in (args.negatives or [])]
    rows = []
    for preset in args.presets:
        deg = PRESETS[preset]
        for L in args.lengths:
            n = int(L * sr)
            pos, neg = [], []
            for p, x in refs:
                for _ in range(args.per_track):
                    s = rng.integers(0, max(1, len(x) - n))
                    pos.append(_run_query(idx, apply(x[s:s + n], sr, deg, noise_bank, rng), _name(p)))
            for p, x in negs:
                for _ in range(args.per_track):
                    s = rng.integers(0, max(1, len(x) - n))
                    neg.append(_run_query(idx, apply(x[s:s + n], sr, deg, noise_bank, rng), None))
            row = {"preset": preset, "len_s": L, **_score(pos)}
            for t in THRESHOLDS:
                row[f"fp@{t}"] = 100 * np.mean([r["score"] >= t for r in neg]) if neg else float("nan")
            rows.append(row)
            print(_fmt_row(row), flush=True)
    _write(args.out, rows)


def _xcorr_lag(ref: np.ndarray, region: np.ndarray) -> int:
    """Sample lag at which `ref` best matches inside `region` (normalised FFT cross-correlation)."""
    n = len(ref) + len(region)
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(region, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)[: len(region) - len(ref) + 1]
    cs = np.r_[0.0, np.cumsum(region**2)]
    energy = np.sqrt(np.maximum(cs[len(ref):] - cs[: len(cs) - len(ref)], 0)) + 1e-9
    return int(np.argmax(c / energy))


def song_spans(audio: np.ndarray, sr: int, songs: list[tuple[str, np.ndarray]], search_s: float = 8.0) -> list[tuple[str, float, float]]:
    """Locate each back-to-back song in a long recording.

    Playback leaves short gaps between songs, so cumulative durations drift.
    Each song start is refined by cross-correlating 20 s taken from the middle
    of the song (intros are often quiet) with the recording near the expected
    position. This is independent of the fingerprinter under test.
    """
    spans, expected = [], 0.0
    for name, x in songs:
        ref_at = 30.0 if len(x) > 60 * sr else 0.0
        ref = x[int(ref_at * sr): int((ref_at + 20) * sr)].astype(np.float64)
        lo = max(0, int((expected + ref_at - search_s) * sr))
        hi = min(len(audio), int((expected + ref_at + search_s) * sr) + len(ref))
        start = (lo + _xcorr_lag(ref, audio[lo:hi].astype(np.float64))) / sr - ref_at
        spans.append((name, start, start + len(x) / sr))
        expected = start + len(x) / sr
    return spans


def recording(args) -> None:
    idx = _load(args)
    sr = idx.cfg.sr
    with open(args.order) as f:
        order = [ln.strip() for ln in f if ln.strip()]
    songs = [(_name(fn), load_audio(os.path.join(args.songs_dir, fn), sr)) for fn in order]
    if args.as_negatives:
        # Recorded songs are removed from the index: every query is now a
        # "song not in the catalogue", so wrong@T is the real-recording false-positive rate.
        idx.remove_tracks({n for n, _ in songs})
    frame_sec = idx.cfg.hop / sr
    rows = []
    dump_f = open(args.dump, "w", newline="") if args.dump else None
    dump = csv.writer(dump_f) if dump_f else None
    if dump:
        dump.writerow(["recording", "len_s", "truth", "pred", "correct", "aligned", "score", "ms"])
    for rec in args.recordings:
        audio = load_audio(rec, sr)
        spans = song_spans(audio, sr, songs)
        for L in args.lengths:
            res = []
            for name, s0, s1 in spans:
                # Non-overlapping windows; skip 1 s at each boundary.
                t = s0 + 1.0
                while t + L <= s1 - 1.0:
                    q = audio[int(t * sr): int((t + L) * sr)]
                    r = _run_query(idx, q, name)
                    # A hit must also land at the right position inside the song.
                    r["aligned"] = r["correct"] and r["offset"] is not None and abs(r["offset"] * frame_sec - (t - s0)) < 0.5
                    if dump is not None:
                        dump.writerow([_name(rec), L, name, r["pred"], int(r["correct"]), int(r["aligned"]),
                                       r["score"], round(r["ms"], 2)])
                    res.append(r)
                    t += L
            row = {"recording": _name(rec), "len_s": L, **_score(res),
                   "aligned_top1": 100 * np.mean([r["aligned"] for r in res])}
            rows.append(row)
            print(_fmt_row(row), flush=True)
    if dump_f:
        dump_f.close()
    _write(args.out, rows)


def _fmt_row(row: dict) -> str:
    return "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())


def _write(path: str | None, rows: list[dict]) -> None:
    if not path:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(prog="afp.evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--density", type=float, default=LandmarkConfig.peaks_per_sec)
    b.add_argument("--fanout", type=int, default=LandmarkConfig.fanout)
    b.add_argument("--method", choices=["landmark", "triangle"], default="landmark")
    b.add_argument("refs", nargs="+")
    b.set_defaults(fn=build)

    s = sub.add_parser("synth")
    s.add_argument("--index", required=True)
    s.add_argument("--refs", nargs="+", required=True)
    s.add_argument("--negatives", nargs="*")
    s.add_argument("--noise", nargs="*")
    s.add_argument("--presets", nargs="+", default=list(PRESETS))
    s.add_argument("--lengths", nargs="+", type=float, default=[1, 2, 3, 5])
    s.add_argument("--per-track", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--distractors", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(fn=synth)

    r = sub.add_parser("recording")
    r.add_argument("--index", required=True)
    r.add_argument("--recordings", nargs="+", required=True)
    r.add_argument("--order", required=True)
    r.add_argument("--songs-dir", required=True)
    r.add_argument("--lengths", nargs="+", type=float, default=[1, 2, 3, 5])
    r.add_argument("--distractors", type=int, default=0)
    r.add_argument("--mode", choices=["o", "a", "f"], help="triangle index query mode")
    r.add_argument("--dump", help="write one row per query (for equal-false-positive comparisons)")
    r.add_argument("--as-negatives", action="store_true",
                   help="remove the recorded songs from the index and measure false positives")
    r.add_argument("--out")
    r.set_defaults(fn=recording)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
