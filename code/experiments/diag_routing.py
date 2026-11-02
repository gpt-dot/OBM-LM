"""diag_routing.py —— 检查点路由诊断：合同市场是否按概念签名把输入路由到专才？

对一个训练完的检查点，逐域对比三种推理方式：
  1. routed   : mesh.eval_ppl —— 线上同款逻辑（cond_reliability + 合同池 + 集成权重）
  2. allcells : 全细胞等权集成（对照：放弃路由）
  3. best1    : 单细胞最优 ppl（对照：理想专才上界）
并打印每域实际被选中的细胞与 ensemble 权重。

论断：若 routed ≈ best1 << allcells，说明路由把每域输入送到了对的细胞——
合同市场不只是记账，是真的在执行分工。
"""
import os, sys, json, argparse, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from obm_lm.lm_mesh import OBMLMConfig
from obm_lm import corpus
from obm_lm.tokenizer import BPETokenizer
from obm_lm import backend as bk
from run_scale import SCALES, build_mesh

CACHE = os.path.join(_ROOT, "obm_mvp/data/cache")
UNIQ = ["en_lit", "zh_classic", "code_py", "en_wiki"]


def ppl_of(mesh, sel, w, seq, tgt, ctx, bs=12):
    """集成 ppl/acc（sel/w 给定，分批 4 控内存）。"""
    nll_sum, n_tok, ok = 0.0, 0, 0
    for b0 in range(0, len(seq), bs):
        s_, t_ = seq[b0:b0 + bs], tgt[b0:b0 + bs]
        for c0 in range(0, len(s_), 4):
            nll, n, c = bk.ens_nll(mesh.cells, sel, w, s_[c0:c0 + 4],
                                   t_[c0:c0 + 4])
            nll_sum += nll; n_tok += n; ok += c
    return float(np.exp(nll_sum / max(n_tok, 1))), ok / max(n_tok, 1)


def selection_of(mesh, seq, ctx):
    """复现 eval_ppl 的选路：返回每 4 序列批的 (选中细胞, 权重)。"""
    out = []
    for c0 in range(0, len(seq), 4):
        s_ = seq[c0:c0 + 4]
        ctxd = mesh._context(mesh.cells[0].signature(s_))
        rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
        sel, _ = mesh._select(rels, allow_reflex=False)
        w = np.exp(mesh.cfg.ensemble_beta * rels[sel])
        w = w / w.sum()
        out.append((list(sel), w, rels))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(SCALES))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    mesh = build_mesh(args.scale, args.seed, tok.vocab_size)
    with open(args.ckpt, "rb") as f:
        mesh.restore(pickle.load(f))
    ctx = mesh.cfg.max_len
    nc = len(mesh.cells)
    print(f"== diag {args.scale} @ step {mesh.step} | cells={nc} "
          f"params={mesh.n_params/1e6:.1f}M")
    print("   contract_floor=%.2f specialist_line=%.2f beta=%.1f"
          % (mesh.cfg.contract_floor, mesh.cfg.specialist_line,
             mesh.cfg.ensemble_beta))

    rows = []
    for k in UNIQ:
        seq, tgt = streams[k].eval_windows(n=24, ctx=ctx)
        # routed
        ppls, accs = [], []
        for b in range(0, len(seq), 12):
            p, a = mesh.eval_ppl(seq[b:b + 12], tgt[b:b + 12],
                                 np.ones(ctx, dtype=bool))
            ppls.append(p); accs.append(a)
        routed_ppl, routed_acc = float(np.mean(ppls)), float(np.mean(accs))
        # all-cells equal weight
        all_ppl, all_acc = ppl_of(mesh, list(range(nc)),
                                  np.ones(nc) / nc, seq, tgt, ctx)
        # per-cell
        per = []
        for i in range(nc):
            p, a = ppl_of(mesh, [i], [1.0], seq, tgt, ctx)
            per.append((p, a))
        best_i = int(np.argmin([p for p, _ in per]))
        # selection
        sels = selection_of(mesh, seq, ctx)
        sel_sum = {}
        w_sum = {}
        for sel, w, rels in sels:
            for j, i in enumerate(sel):
                sel_sum[i] = sel_sum.get(i, 0) + 1
                w_sum[i] = w_sum.get(i, 0.0) + float(w[j])
        nsel = len(sels)
        sel_str = " ".join(
            f"c{i}:{sel_sum.get(i,0)}/{nsel}(w={w_sum.get(i,0.0)/nsel:.2f})"
            for i in range(nc) if sel_sum.get(i, 0))
        rows.append(dict(domain=k, routed=routed_ppl, allcells=all_ppl,
                         best1=per[best_i][0], best_cell=best_i,
                         per_cell=[round(p, 1) for p, _ in per],
                         routed_acc=routed_acc))
        print(f"\n[{k}] routed={routed_ppl:12.1f}  all={all_ppl:12.1f}  "
              f"best1=c{best_i}:{per[best_i][0]:12.1f}")
        print("   per-cell ppl: " + " ".join(
            f"c{i}={p:.1f}" for i, (p, _) in enumerate(per)))
        print(f"   selected: {sel_str if sel_str else '(none)'}")

    # 领地总览
    print("\n== territory (seg_rel / seg_best)")
    for i, c in enumerate(mesh.cells):
        print(f"  c{i}: rel={ {k: round(v,3) for k,v in c.seg_rel.items()} } "
              f"best={ {k: round(v,3) for k,v in c.seg_best.items()} }")

    out = dict(scale=args.scale, step=mesh.step, cells=nc, rows=rows,
               territory=[dict(cell=i,
                               seg_rel={str(k): round(v, 3) for k, v in c.seg_rel.items()},
                               seg_best={str(k): round(v, 3) for k, v in c.seg_best.items()})
                          for i, c in enumerate(mesh.cells)])
    dst = os.path.join(os.path.dirname(args.ckpt), "diag_routing.json")
    with open(dst, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n== saved {dst}")


if __name__ == "__main__":
    main()
