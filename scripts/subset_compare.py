"""Compare all methods on exactly the same SD-RR queries (1/3 subset, window start 0, 3 s and 10 s)."""
import csv, sys
import numpy as np
sys.path.insert(0, "/home/user/audio-fingerprint")
from afp.compare import operating_point

R = "/home/user/afp-data/runs"
S = "/home/user/afp-data/sdrr"
sub = {r["query_id"] for r in csv.DictReader(open(f"{S}/manifest_sub3.csv"))}

def rows(path):
    seen, out = set(), []
    for r in csv.DictReader(open(path)):
        L = float(r["len_s"])
        key = (r["recording"], L)
        if r["recording"] not in sub or L not in (3.0, 10.0) or key in seen:
            continue                      # keep the first window (start 0 s) only
        seen.add(key); out.append(r)
    return out

methods = {"baseline (pair hash)": "dump_sdrr_landmark.csv", "A (triangle, POLARIS)": "dump_sdrr_triangle.csv",
           "B (neural, NMFP)": "dump_sdrr_neural_sub.csv"}
print(f"{'method':24s} {'len':>4s} {'n':>4s} {'top1':>6s} {'acc@FP1%':>9s} {'acc@FP0.1%':>11s} {'ms':>6s}")
out = []
for name, f in methods.items():
    rs = rows(f"{R}/{f}")
    for L in (3.0, 10.0):
        g = [r for r in rs if float(r["len_s"]) == L]
        sc = np.array([float(r["score"]) for r in g]); im = np.array([float(r["imp"]) for r in g])
        ok = np.array([r["correct"] == "1" for r in g])
        a1, _ = operating_point(sc, im, ok, 1.0); a01, _ = operating_point(sc, im, ok, 0.1)
        ms = np.median([float(r["ms"]) for r in g])
        print(f"{name:24s} {L:4.0f} {len(g):4d} {100*ok.mean():6.1f} {a1:9.1f} {a01:11.1f} {ms:6.0f}")
        out.append(dict(method=name, len_s=L, n=len(g), top1=round(100*ok.mean(), 1), acc_fp1=round(a1, 1),
                        acc_fp01=round(a01, 1), median_ms=round(ms)))
with open(f"{R}/compare_sdrr_sub.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
