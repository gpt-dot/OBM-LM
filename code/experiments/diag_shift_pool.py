"""聚焦诊断：shift 段（1800-2000）逐步打印 ctx / rels / 合同池 / 选中细胞。

验证假设：copy 专才 cell #0 因 shift 输入的 ctx 在段#0(copy)上权重 ≥0.11，
cond_rel = 0.5 + 0.45*w0 ≥ 0.55 跨过合同线，被拉进 shift 池以全额 lr 训练。
"""
import os, sys

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # 位级可复现：固定 BLAS 线程（见 obm_lm/__init__.py）

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm.langs import build_blocks

BATCH = 12


def main():
    cfg = OBMLMConfig(seed=1)
    model = OBMLM(cfg)
    blocks, test_sets, names = build_blocks(0, 160)

    # 前 3 段静默跑完（与主实验同一条流）
    for bi in range(3):
        for _ in range(600):
            seq, tgt, mask = blocks[bi].batch(BATCH)
            model.run_step(seq, tgt, mask)
    print("=== shift 段开始前的细胞账本 ===")
    for c in model.cells:
        print(f"  cell #{c.cell_id}: seg_rel="
              f"{ {k: round(v, 3) for k, v in c.seg_rel.items()} } "
              f"seg_best={ {k: round(v, 3) for k, v in c.seg_best.items()} }")
    print(f"  segments: {[s['id'] for s in model.segments]} "
          f"seg_counter={model.seg_counter}")

    # 插桩 run_step：shift 段每步打印
    orig_run = model.run_step

    def run_spy(seq, tgt, mask):
        sig = model.cells[0].signature(seq)
        for c in model.cells:
            c.erode(model.seg_counter, model.step, cfg.inactivity_decay,
                    cfg.commitment_decay)
        model._update_sig_stats(sig)
        ctx = model._context(sig)
        rels = np.array([c.cond_reliability(ctx) for c in model.cells])
        commitments = np.array([max(c.seg_best.values(), default=0.0)
                                for c in model.cells])
        pool = model._valid(rels, probe=None, commitments=commitments,
                            force_award=False)
        d = orig_run(seq, tgt, mask)
        st = model.step - 1
        if 1800 <= st <= 1960:
            ctx_txt = ", ".join(f"#{k}:{v:.2f}" for k, v in
                                sorted(ctx.items(), key=lambda kv: -kv[1])[:4])
            print(f"  step {st}: d_last={model.d_last:.2f} d_run={model.d_run} "
                  f"drop_run={model.drop_run} | ctx[{ctx_txt}] "
                  f"rels={[round(float(r), 3) for r in rels]} "
                  f"pool={pool} sel={d['selected']} "
                  f"accs={[round(a, 2) for a in d['cell_accs']]} "
                  f"ens={d['ens_acc']:.2f}", flush=True)
        return d

    model.run_step = run_spy
    for _ in range(200):
        seq, tgt, mask = blocks[3].batch(BATCH)
        model.run_step(seq, tgt, mask)
    print("=== shift 段 200 步后的细胞账本 ===")
    for c in model.cells:
        print(f"  cell #{c.cell_id}: seg_rel="
              f"{ {k: round(v, 3) for k, v in c.seg_rel.items()} } "
              f"seg_best={ {k: round(v, 3) for k, v in c.seg_best.items()} }")


if __name__ == "__main__":
    main()
