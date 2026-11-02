"""model_soup.py —— 模型汤实验（头脑风暴 #2）。

把多代权重逐细胞（按 cell_id 匹配）平均，测"免费合体"是否成立：
  soup(A,B) = (wA + wB) / 2，逐参数张量。
评估口径：market eval_ppl + NLL router top1/adaptive，窗口子采样加速。

用法：python3 experiments/model_soup.py A.pkl B.pkl [C.pkl]
"""
import os, sys, pickle, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from obm_lm.lm_mesh import OBMLM
from obm_lm import corpus, backend as bk
from obm_lm.tokenizer import BPETokenizer

CACHE = os.path.join(os.environ.get("OBM_ROOT", os.path.join(_ROOT, "obm_mvp")),
                     "data", "cache")


def load_raw(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_soup(paths):
    raws = [load_raw(p) for p in paths]
    base = raws[0]
    # cell_id -> params
    per_cell = {}
    for r in raws:
        for c in r["cells"]:
            per_cell.setdefault(c["cell_id"], []).append(c["params"])
    common = sorted(cid for cid, ps in per_cell.items() if len(ps) == len(raws))
    print(f"[soup] sources={len(raws)} common_cells={common}")
    soup_cells = []
    for cid in common:
        ps = per_cell[cid]
        avg = {}
        for k in ps[0]:
            arrs = [np.asarray(p[k], dtype=np.float32) for p in ps]
            m = np.mean(arrs, axis=0)
            avg[k] = m.astype(ps[0][k].dtype)
        soup_cells.append({"cell_id": cid, "params": avg,
                           **{k: v for k, v in base["cells"][0].items()
                              if k not in ("cell_id", "params")}})
    out = dict(base)
    out["cells"] = soup_cells
    out["step"] = max(r.get("step", 0) for r in raws)
    out["soup_of"] = [os.path.basename(p) for p in paths]
    return out, base


def save_soup(soup, path):
    with open(path, "wb") as f:
        pickle.dump(soup, f, protocol=4)
    print(f"[soup] saved {path} {os.path.getsize(path)/1e6:.0f}MB")


def eval_mesh(mesh, streams, max_win=48, ctx=None):
    ctx = ctx or mesh.cfg.max_len
    res = {}
    for k in corpus.DOMAINS:
        seq, tgt = streams[k].eval_windows(ctx=ctx)
        seq, tgt = seq[:max_win], tgt[:max_win]
        # market
        ppls = []
        for b in range(0, len(seq), 12):
            p, _ = mesh.eval_ppl(seq[b:b+12], tgt[b:b+12],
                                 np.ones(ctx, dtype=bool))
            ppls.append(p)
        # per-cell full-window NLL -> router
        nlls = []
        for c in mesh.cells:
            vals = [bk.mean_nll(c, seq[b:b+12], tgt[b:b+12])
                    for b in range(0, len(seq), 12)]
            nlls.append(float(np.mean(vals)))
        nlls = np.array(nlls)
        top1 = float(np.exp(nlls.min()))
        # adaptive blend（tau=0.15，全窗口软权重）
        order = np.argsort(nlls)
        w = np.zeros(len(mesh.cells))
        w[order[0]] = 1.0
        if len(mesh.cells) > 1 and nlls[order[1]] - nlls[order[0]] < 0.15:
            w[order[1]] = np.exp(-4.0 * (nlls[order[1]] - nlls[order[0]]))
            w /= w.sum()
        sel = [i for i in range(len(mesh.cells)) if w[i] > 0.005]
        tot = 0.0; ntok = seq.size
        for b in range(0, len(seq), 12):
            n, _, _ = bk.ens_nll(mesh.cells, sel,
                                 [w[i] for i in sel], seq[b:b+12], tgt[b:b+12])
            tot += n
        res[k] = dict(market=round(float(np.mean(ppls)), 0),
                      router_top1=round(top1, 0),
                      router_ada=round(float(np.exp(tot/ntok)), 0),
                      best_cell=int(order[0]))
    return res


def main():
    paths = sys.argv[1:]
    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    soup, base = make_soup(paths)
    out_path = "/tmp/soup_weights.pkl"
    save_soup(soup, out_path)

    mesh = OBMLM.__new__(OBMLM)
    mesh.__init__(soup["cfg"])
    mesh.load_weights_only(out_path, param_budget_bytes=int(5.5e9))
    print(f"[soup] cells={[c.cell_id for c in mesh.cells]} "
          f"params={mesh.n_params/1e6:.1f}M step={mesh.step}")
    t0 = time.time()
    res = eval_mesh(mesh, streams)
    print(f"[soup] eval {time.time()-t0:.0f}s")
    print(f"{'domain':12s} {'market':>8s} {'router1':>8s} {'router_ada':>10s} cell")
    for k, v in res.items():
        print(f"{k:12s} {v['market']:>8.0f} {v['router_top1']:>8.0f} "
              f"{v['router_ada']:>10.0f}  #{mesh.cells[v['best_cell']].cell_id}")


if __name__ == "__main__":
    main()
