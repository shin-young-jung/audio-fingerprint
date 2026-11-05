import numpy as np

from afp.degrade import PRESETS, apply
from afp.index import LandmarkIndex
from afp.landmark import LandmarkConfig, fingerprint, pack_hash, unpack_hash

SR = 8000


def synth_track(seed: int, seconds: float = 20.0) -> np.ndarray:
    """Random melody of short tones: distinct per seed, peaky like music."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(seconds / 0.25)):
        t = np.arange(int(0.25 * SR)) / SR
        f0 = rng.uniform(150, 1500)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
        out.append(tone * np.hanning(len(t)))
    x = np.concatenate(out)
    return (x / np.abs(x).max()).astype(np.float32)


def test_hash_roundtrip():
    f1 = np.array([0, 10, 511]); f2 = np.array([511, 3, 0]); dt = np.array([1, 63, 32])
    a, b, c = unpack_hash(pack_hash(f1, f2, dt))
    assert (a == f1).all() and (b == f2).all() and (c == dt).all()


def test_fingerprint_density_reasonable():
    h, t = fingerprint(synth_track(0), LandmarkConfig())
    per_sec = len(h) / 20.0
    assert 20 < per_sec < 400
    assert (np.diff(t.astype(np.int64)) >= 0).all()


def build(n=5):
    idx = LandmarkIndex()
    tracks = [synth_track(s) for s in range(n)]
    for i, x in enumerate(tracks):
        idx.add(f"t{i}", x)
    idx.finalize()
    return idx, tracks


def test_exact_excerpt_matches_with_offset():
    idx, tracks = build()
    start = 5.0
    q = tracks[3][int(start * SR): int((start + 3) * SR)]
    m = idx.query(q)
    assert m[0].name == "t3"
    assert abs(m[0].offset_frames * idx.cfg.hop / SR - start) < 0.1
    assert m[0].score > m[0].second_score


def test_degraded_excerpt_still_matches():
    idx, tracks = build()
    rng = np.random.default_rng(1)
    q = apply(tracks[2][SR * 4: SR * 9], SR, PRESETS["noise_10db"], [], rng)
    assert idx.query(q)[0].name == "t2"


def test_save_load(tmp_path):
    idx, tracks = build(3)
    p = tmp_path / "i.npz"
    idx.save(str(p))
    idx2 = LandmarkIndex.load(str(p))
    assert idx2.names == idx.names and len(idx2) == len(idx)
    assert idx2.query(tracks[1][SR * 2: SR * 5])[0].name == "t1"


def test_synthetic_distractors_keep_true_match():
    idx, tracks = build(3)
    n_real = len(idx)
    idx.add_synthetic_distractors(50, sec_per_track=20)
    assert len(idx.names) == 53 and len(idx) > n_real * 10
    assert idx.query(tracks[0][SR * 3: SR * 6])[0].name == "t0"


def test_remove_tracks_turns_query_into_negative():
    idx, tracks = build(3)
    idx.remove_tracks({"t1"})
    m = idx.query(tracks[1][SR * 2: SR * 5])
    assert not m or m[0].name != "t1"


def test_triangle_index_finds_excerpt_and_offset():
    from afp.triangle import TriangleIndex
    idx = TriangleIndex()
    tracks = [synth_track(s) for s in range(4)]
    for i, x in enumerate(tracks):
        idx.add(f"t{i}", x)
    idx.finalize()
    m = idx.query(tracks[2][SR * 5: SR * 8])
    assert m[0].name == "t2"
    assert abs(m[0].offset_frames * idx.cfg.hop / SR - 5.0) < 0.1
