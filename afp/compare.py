"""Compare methods at the same false-positive rate.

  python -m afp.compare NAME=dump.csv [NAME=dump.csv ...] [--fp 1 0.1]

Each dump row is one query with `score` (best candidate) and `imp` (best
candidate that is NOT the true song). Scores are not comparable across methods
(hash counts vs cosine similarity), so for each method we pick the threshold at
which at most F% of queries would accept a wrong song if their true song were
missing from the catalogue (imp > threshold), and report how many queries are
then correctly accepted (correct and score > threshold).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np


def operating_point(score: np.ndarray, imp: np.ndarray, correct: np.ndarray, fp_pct: float) -> tuple[float, float]:
    """Return (accept %, threshold) at false-positive rate <= fp_pct."""
    n = len(imp)
    k = int(np.floor(fp_pct / 100 * n))          # impostors allowed above threshold
    desc = np.sort(imp)[::-1]
    thr = desc[k] if k < n else -np.inf          # accept only scores strictly above this
    return 100 * float(np.mean(correct & (score > thr))), float(thr)


def load(path: str) -> dict:
    groups: dict = defaultdict(lambda: {"score": [], "imp": [], "correct": [], "aligned": [], "ms": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["recording"], float(r["len_s"]))
            g = groups[key]
            g["score"].append(float(r["score"])); g["imp"].append(float(r["imp"]))
            g["correct"].append(r["correct"] == "1"); g["aligned"].append(r["aligned"] == "1")
            g["ms"].append(float(r["ms"]))
    return {k: {kk: np.array(v) for kk, v in g.items()} for k, g in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser(prog="afp.compare")
    ap.add_argument("dumps", nargs="+", help="NAME=path.csv")
    ap.add_argument("--fp", nargs="+", type=float, default=[1.0, 0.1])
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = []
    for spec in args.dumps:
        name, path = spec.split("=", 1)
        for (rec, L), g in sorted(load(path).items()):
            row = {"method": name, "set": rec, "len_s": L, "n": len(g["score"]),
                   "top1": round(100 * g["correct"].mean(), 1)}
            for fp in args.fp:
                acc, thr = operating_point(g["score"], g["imp"], g["correct"], fp)
                row[f"accept@fp{fp:g}%"] = round(acc, 1)
            row["median_ms"] = round(float(np.median(g["ms"])), 1)
            rows.append(row)

    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in sorted(rows, key=lambda r: (r["set"], r["len_s"], r["method"])):
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
