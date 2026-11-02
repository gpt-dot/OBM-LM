"""torch_smoke.py —— torch 路全机制冒烟：训练步 + 梦境代谢 + eval_ppl + 生成。

prof_step 只覆盖了 run_step 主路径；本脚本用玩具协议 + 短梦境周期，
把 dream()（含分裂选型/凋亡/合同核对）、eval_ppl、checkpoint 全跑一遍，
确认新增的引擎分派助手（ens_acc/ens_nll/argmax_pred/mean_nll/
forward_stats 融合版）在 torch 路无崩溃且数值合理。
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(8)

from obm_lm import OBMLM, OBMLMConfig, build_blocks, PLAN

BLOCK, BATCH = 80, 12


def main():
    blocks, test_sets, names = build_blocks(0, 48)
    test_pairs = [(s, t) for s, t, _ in test_sets]
    masks = [m for _, _, m in test_sets]
    # 玩具协议必须用玩具标定超参（acc_prior=0.34/fulfill 0.30/expert 0.65/
    # warmup 250/cooldown 120）——规模标定值（acc_prior=0 + 低门限）是给
    # 真实域的超远签名距离准备的，玩具任务签名近，三种后端下都零事件。
    # 仅把 warmup/cooldown 压缩以便 480 步内看到机制全貌。
    cfg = OBMLMConfig(seed=3, backend="torch", vocab=16,
                      d_model=64, n_heads=4, n_layers=2, max_len=14,
                      n_cells_init=4, max_cells=6,
                      out_full=True, amp_bf16=True,
                      grad_clip=1.0, weight_decay=0.0,
                      lr=0.006, lr_breach=0.002, dream_lr=0.006,
                      dream_batches=4, dream_batch=12, dream_period=40,
                      buffer_size=600, seg_buffer_size=60,
                      dist_refresh=10, gauss_chi=0.0,
                      warmup=40, boundary_cooldown=30)
    model = OBMLM(cfg)
    t0 = time.time()
    step = 0
    for bi, block in enumerate(blocks):
        for _ in range(BLOCK):
            seq, tgt, mask = block.batch(BATCH)
            rec = model.run_step(seq, tgt, mask)
            step += 1
        ppl, acc = model.eval_ppl(seq, tgt, mask)
        print(f"[block {bi}] steps={step} ens_acc={rec['ens_acc']:.3f} "
              f"cells={len(model.cells)} eval_acc={acc:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    from collections import Counter
    cnt = Counter(t for _, t, _ in model.events)
    print("\n== 事件 ==", dict(cnt))
    for st, t, d in model.events:
        print(f"  [{st:5d}] {t:6s} {d}")

    # checkpoint 往返
    ck = model.checkpoint()
    m2 = OBMLM(cfg)
    m2.restore(ck)
    seq, tgt, mask = blocks[0].batch(BATCH)
    r1 = model.run_step(seq, tgt, mask)
    r2 = m2.run_step(seq, tgt, mask)
    same = abs(r1["ens_acc"] - r2["ens_acc"]) < 1e-9
    print(f"\ncheckpoint 往返一致: {same} ({r1['ens_acc']:.6f} vs {r2['ens_acc']:.6f})")

    # 生成（各细胞集成路由贪心解码，与 run_scale.generate 同逻辑）
    from obm_lm import backend as bk
    ids = list(seq[0][:12])
    for _ in range(24):
        window = np.asarray([ids[-14:]], dtype=np.int64)
        ctxd = model._context(model.cells[0].signature(window))
        rels = np.array([c.cond_reliability(ctxd) for c in model.cells])
        sel, _ = model._select(rels, allow_reflex=False)
        ps = []
        for i in sel:
            lg = bk.forward(model.cells[i], window)
            p = np.exp(lg - lg.max(-1, keepdims=True))
            ps.append((p / p.sum(-1, keepdims=True))[0, -1])
        w = np.exp(model.cfg.ensemble_beta * rels[sel])
        w /= w.sum()
        dist = sum(wi * p for wi, p in zip(w, ps))
        ids.append(int(np.argmax(dist)))
    print("\n== 生成样本（token ids） ==", ids[:40])
    print("\n[SMOKE OK]" if same else "\n[SMOKE FAIL] checkpoint")


if __name__ == "__main__":
    main()
