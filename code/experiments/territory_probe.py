"""territory_probe.py —— 细胞领地测量探针（TRIZ：理想最终结果的度量反馈）。

空间分离"声称"每个域有自己的细胞领地，但从未直接测量。本探针对每个域的
评估窗口逐序列计算路由（与 eval_ppl 同路径：signature -> _context ->
cond_reliability -> _select），统计 域×细胞 的选择份额与平均可靠度。

用法：python3 experiments/territory_probe.py --weights <pkl> [多个]
"""
import os, sys, json, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from obm_lm.lm_mesh import OBMLM
from obm_lm import corpus
from obm_lm.tokenizer import BPETokenizer

_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_OBM_ROOT, "data", "cache")


def load_mesh(path, budget_gb=5.5):
    with open(path, "rb") as f:
        d = pickle.load(f)
    mesh = OBMLM.__new__(OBMLM)
    mesh.__init__(d["cfg"])
    mesh.load_weights_only(path, param_budget_bytes=int(budget_gb * 1e9))
    return mesh


def probe(mesh, streams, ctx=128):
    out = {}
    for k in corpus.DOMAINS:
        seq, _ = streams[k].eval_windows(ctx=ctx)
        n_cells = len(mesh.cells)
        sel_cnt = np.zeros(n_cells)
        rel_sum = np.zeros(n_cells)
        n_seq = 0
        for i in range(len(seq)):
            s_ = seq[i:i + 1]
            ctxd = mesh._context(mesh.cells[0].signature(s_))
            rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
            sel, _ = mesh._select(rels, allow_reflex=False)
            for j in sel:
                sel_cnt[j] += 1
            rel_sum += rels
            n_seq += 1
        ids = [c.cell_id for c in mesh.cells]
        out[k] = {
            "cell_ids": ids,
            "sel_share": (sel_cnt / max(1, sel_cnt.sum())).round(3).tolist(),
            "mean_rel": (rel_sum / max(1, n_seq)).round(3).tolist(),
            "n_seq": n_seq,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    args = ap.parse_args()
    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    all_out = {}
    for w in args.weights:
        tag = os.path.basename(w).split("-")[0]
        mesh = load_mesh(w)
        print(f"[{tag}] cells={[c.cell_id for c in mesh.cells]} "
              f"step={mesh.step}", flush=True)
        r = probe(mesh, streams)
        all_out[tag] = r
        for k, v in r.items():
            share = " ".join(f"#{i}:{s:.2f}" for i, s in zip(v["cell_ids"], v["sel_share"]))
            rel = " ".join(f"#{i}:{s:.2f}" for i, s in zip(v["cell_ids"], v["mean_rel"]))
            print(f"  {k:10s} sel[{share}]  rel[{rel}]", flush=True)
    with open(os.path.join(_OBM_ROOT, "results_lab", "territory_probe.json"), "w") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=1)
    print("[done] -> results_lab/territory_probe.json", flush=True)


if __name__ == "__main__":
    main()
