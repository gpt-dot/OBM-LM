import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from obm_lm.lm_mesh import OBMLM
from obm_lm import backend as bk
from obm_lm.tokenizer import BPETokenizer
import pickle

W = sys.argv[1] if len(sys.argv) > 1 else "results_lab/scale2_local_weights.pkl"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
print("loading", W, flush=True)
tok = BPETokenizer.load("data/cache/tokenizer.json")
m = OBMLM.__new__(OBMLM)
m.__init__(pickle.load(open(W, "rb"))["cfg"])
m.load_weights_only(W)
print("loaded cells", len(m.cells), "ctx", m.cfg.max_len, flush=True)

ids = tok.encode("def ")
ids = ids[-(m.cfg.max_len - 1):]
while len(ids) < m.cfg.max_len:
    ids = [0] + ids
seq = np.array([ids], dtype=np.int64)
all_t = time.time()
for step in range(GEN):
    t = time.time()
    ctxd = m._context(m.cells[0].signature(seq))
    rels = np.array([c.cond_reliability(ctxd) for c in m.cells])
    sel, _ = m._select(rels, allow_reflex=False)
    w = np.exp(m.cfg.ensemble_beta * rels[sel]); w = w / w.sum()
    lg = None
    for j, ci in enumerate(sel):
        g = bk.forward(m.cells[ci], seq, train=False)
        if lg is None:
            lg = np.zeros_like(g)
        lg = lg + w[j] * g
    nxt = int(np.argmax(lg[0, -1]))
    ids = ids + [nxt]
    ids = ids[-m.cfg.max_len:]
    seq = np.array([ids], dtype=np.int64)
    print(f"  tok{step} sel={sel} nxt={nxt} ({time.time()-t:.2f}s) -> {tok.decode([nxt])!r}", flush=True)
print("TOTAL", time.time() - all_t, flush=True)
print("FINAL:", tok.decode(ids), flush=True)
