import os, sys, time
os.environ["OBM_THREADS"] = "4"
import numpy as np
sys.path.insert(0, ".")
from obm_lm.lm_mesh import OBMLM
from obm_lm import backend as bk
from obm_lm.tokenizer import BPETokenizer
import pickle

CACHE = "data/cache"
tok = BPETokenizer.load(f"{CACHE}/tokenizer.json")
mesh = OBMLM.__new__(OBMLM)
d = pickle.load(open("results_lab/scale2_local_weights.pkl", "rb"))
mesh.__init__(d["cfg"])
mesh.load_weights_only("results_lab/scale2_local_weights.pkl")
print("loaded", len(mesh.cells), "cells", flush=True)
ctx = mesh.cfg.max_len
ids = tok.encode("def ")
ids = ids[-(ctx - 1):]
while len(ids) < ctx:
    ids = [0] + ids
ids = ids[-ctx:]
seq = np.array([ids], dtype=np.int64)
print("seq", seq.shape, "ctx", ctx, flush=True)

def step(name, fn):
    print("--> " + name, flush=True)
    t = time.perf_counter()
    r = fn()
    print(f"<-- {name} {time.perf_counter()-t:.3f}s", flush=True)
    return r

sig = step("signature", lambda: mesh.cells[0].signature(seq))
ctxd = step("_context", lambda: mesh._context(sig))
print("ctxd keys", list(ctxd.keys())[:5], flush=True)
rels = step("cond_reliability", lambda: np.array([c.cond_reliability(ctxd) for c in mesh.cells]))
sel, info = step("_select", lambda: mesh._select(rels, allow_reflex=False))
print("sel", sel, "rels", np.round(rels, 3), flush=True)
for j, ci in enumerate(sel):
    lg = step(f"forward cell{ci}", lambda c=ci: bk.forward(mesh.cells[c], seq, train=False))
    print("  shape", lg.shape, flush=True)
print("DONE-STEP1", flush=True)
