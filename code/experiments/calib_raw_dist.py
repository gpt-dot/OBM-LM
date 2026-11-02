"""标定：原始 batch 签名距离（信号 B 快速版）的段内/跨任务分布。

_context 用的是 batch 均值到段高斯的 _gauss_dist（var_floor 作点方差）。
切换首步就要能判别，因此需要知道同一形式下：
  - 段内：batch(12) 均值 -> 本任务冻结参照(240) 的距离分布
  - 跨任务：batch 均值 -> 异任务冻结参照的距离分布
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.langs import build_blocks, TASKS, V
from obm_lm.lm_mesh import OBMLMConfig

cfg = OBMLMConfig()
rng = np.random.default_rng(7)
sig_emb = rng.normal(0.0, 1.0, (V, cfg.d_model))

blocks, test_sets, names = build_blocks(0, 400)


def sig_of(seq):
    return sig_emb[seq].mean(axis=1)


def gauss_dist(x, m, v):
    return float(0.5 * np.sum((x - m) ** 2 / (cfg.var_floor + v)))


# 每个任务：冻结参照（240 条）+ 200 个 batch(12) 的均值
refs, batches = {}, {}
for name in names:
    seq, _, _ = blocks[[b.name for b in blocks].index(name)].testset(400)
    xs = sig_of(seq[:240])
    m, v = xs.mean(axis=0), np.maximum(xs.var(axis=0), cfg.var_floor)
    refs[name] = (m, v)
    bm = [sig_of(seq[240 + 12 * i:252 + 12 * i]).mean(axis=0) for i in range(13)]
    batches[name] = bm

print("=== 段内距离（batch 均值 -> 本任务参照）===")
within = []
for name in names:
    m, v = refs[name]
    ds = [gauss_dist(bm, m, v) for bm in batches[name]]
    within += ds
    print(f"  {name:8s} mean={np.mean(ds):.2f} max={np.max(ds):.2f}")
print(f"  全体: mean={np.mean(within):.2f} p95={np.percentile(within, 95):.2f} "
      f"max={np.max(within):.2f}")

print("=== 跨任务距离（batch 均值 -> 异任务参照）===")
cross = []
for name in names:
    for other in names:
        if other == name:
            continue
        m, v = refs[other]
        ds = [gauss_dist(bm, m, v) for bm in batches[name]]
        cross += ds
        print(f"  {name:8s} -> {other:8s} mean={np.mean(ds):.2f} "
              f"min={np.min(ds):.2f}")
print(f"  全体: mean={np.mean(cross):.2f} min={np.min(cross):.2f} "
      f"p5={np.percentile(cross, 5):.2f}")
