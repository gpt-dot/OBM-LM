"""diag_note.py —— 一次性诊断：逐细胞×域 next-token 精度 + eval 选路明细。

回答两个问题：
 1. 难域（en_wiki/zh）的专精细胞精度是否低于 _note_bar（0.12-0.30）？
    ——若是，note(ok=False) 会把专才 seg_rel 磨到 0，合同永远不成。
 2. eval 时每域的 ctx 权重、各细胞 cond_rel、最终选中与权重——
    验证 cond_reliability 的未知段 0.5 先验是否让未受训细胞压过专才。
"""
import os, sys, pickle
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


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_ROOT, "obm_mvp/results_scale/pilot_v1/ckpt_final.pkl")
    scale = sys.argv[2] if len(sys.argv) > 2 else "pilot"
    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    mesh = build_mesh(scale, 0, tok.vocab_size)
    with open(ckpt, "rb") as f:
        mesh.restore(pickle.load(f))
    ctx_len = mesh.cfg.max_len
    nc = len(mesh.cells)

    print("== per-cell x domain next-token acc / ppl")
    print("   note_bar(fulfill=0.12, expert=0.30): r=0.5->0.12 r=0.9->0.30")
    for k in UNIQ:
        seq, tgt = streams[k].eval_windows(n=8, ctx=ctx_len)
        line = []
        for i in range(nc):
            # forward_stats 直接给 probe acc（与 decide 的 cell_accs 同源）
            fs = mesh.cells[i].forward_stats(seq, np.ones(ctx_len, dtype=bool), tgt)
            line.append(f"c{i}: acc={fs['probe']:.3f}")
        print(f"  [{k}] " + " | ".join(line))

    print("\n== eval selection detail (ctx / rels / sel / w)")
    for k in UNIQ:
        seq, tgt = streams[k].eval_windows(n=8, ctx=ctx_len)
        s_ = seq[:4]
        ctxd = mesh._context(mesh.cells[0].signature(s_))
        rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
        sel, info = mesh._select(rels, allow_reflex=False)
        w = np.exp(mesh.cfg.ensemble_beta * rels[sel])
        w = w / w.sum()
        ctx_str = " ".join(f"s{s}:{wt:.2f}" for s, wt in sorted(ctxd.items()))
        rel_str = " ".join(f"c{i}:{rels[i]:.3f}" for i in range(nc))
        sel_str = " ".join(f"c{i}(w={w[j]:.2f})" for j, i in enumerate(sel))
        print(f"  [{k}] ctx={{{ctx_str}}}")
        print(f"        rels={rel_str}")
        print(f"        sel={sel_str}  info={info if not isinstance(info, dict) else {kk: vv for kk, vv in info.items() if kk != 'rels'}}")


def _seg_of(mesh, domain):
    """域 -> 段 id 映射（按 PLAN 顺序：seg i 对应 PLAN[i] 的首次出现）。"""
    plan = ["en_lit", "zh_classic", "code_py", "en_wiki", "en_lit", "zh_classic"]
    return plan.index(domain)


if __name__ == "__main__":
    main()
