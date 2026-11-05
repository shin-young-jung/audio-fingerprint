"""Reproduce the POLARIS paper's NMFP SD-RR number (Table 2: 74.33% track Top-1) with afp.neural."""
import csv, glob, os, sys, numpy as np, essentia.standard as es
sys.path.insert(0, "/home/user/audio-fingerprint")
from afp.neural import NeuralIndex
S = '/home/user/afp-data/sdrr'
idx = NeuralIndex.load('/home/user/afp-data/runs/idx1k_neural.npz')
sdrr = {os.path.basename(p)[:-4] for p in glob.glob(f'{S}/references/*.mp3')}
idx.remove_tracks({n for n in idx.names if n not in sdrr})          # paper: 496 references only
rows = list(csv.DictReader(open(f'{S}/manifest.csv')))
ok = off = 0
for i, r in enumerate(rows, 1):
    a = es.MonoLoader(filename=f"{S}/{r['query_path']}", sampleRate=8000, resampleQuality=0)().astype(np.float32)
    m = idx.query(a)[0]
    hit = m.name == r['reference_path'].split('/')[-1][:-4]
    ok += hit
    off += hit and abs(m.offset_frames * 0.5 - float(r['reference_begin_seconds'])) <= 0.5
    if i % 200 == 0:
        print(f'{i}/{len(rows)} running top1={100*ok/i:.2f}%', flush=True)
print(f'FINAL n={len(rows)} track top1={100*ok/len(rows):.2f}% (paper NMFP 74.33%), track+offset(0.5s)={100*off/len(rows):.2f}%')
