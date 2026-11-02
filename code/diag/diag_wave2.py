"""完整统计 wave 200 个测试输入的失分模式（对齐 run_all 评估协议：纯 decide，无 observe）。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cfg = OBMConfig(seed=seed)
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(cfg)

for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)

wi = names.index("wave")
Xa, Ya = np.array(test_sets[wi][0]), np.array(test_sets[wi][1])
n = len(Xa)

top1 = 0; c0_only = 0
ctx2 = []; w0 = []; err_sel0 = 0; err_n = 0
sel_size = []; n_sel_with_0 = 0
reflex_n = 0
for i in range(n):
    x = Xa[i]
    p = model.perceive(x)
    pred, conf, selected, dists, ens, reflex, ctx = model.decide(p, count_energy=False)
    ok = int(pred == Ya[i])
    top1 += ok
    reflex_n += int(reflex)
    ctx2.append(dict(ctx).get(2, 0.0))
    if 0 in model.cells:
        c0_only += int(np.argmax(model.cells[0].forward(p)) == Ya[i])
    sel_size.append(len(selected))
    if 0 in selected:
        n_sel_with_0 += 1
        rels = np.array([c.cond_reliability(ctx) for c in model.cells])
        w = np.exp(cfg.ensemble_beta * rels[list(selected)])
        w /= w.sum()
        w0.append(w[list(selected).index(0)])
    if not ok:
        err_n += 1
        if 0 in selected:
            err_sel0 += 1

w0 = np.array(w0); ctx2 = np.array(ctx2); sel_size = np.array(sel_size)
print(f"wave 测试 n={n}  (seed {seed})")
print(f"集成精度             : {top1/n:.3f}")
print(f"#0 单独精度          : {c0_only/n:.3f}")
print(f"反射弧触发比例       : {reflex_n/n:.3f}")
print(f"ctx[#2] mean={ctx2.mean():.3f}  <0.3 占比={np.mean(ctx2<0.3):.3f}  <0.15 占比={np.mean(ctx2<0.15):.3f}")
print(f"#0 被选中比例        : {n_sel_with_0/n:.3f}   被选中时 w0: mean={np.nanmean(w0) if len(w0) else 0:.3f}")
print(f"选中集大小 mean={sel_size.mean():.2f}")
print(f"错误样本 {err_n} 个，其中 #0 被选中 {err_sel0} 个")
