"""细查：ctx[#2] 低的输入是哪些 + 错误样本的选中集构成。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

seed = 1
cfg = OBMConfig(seed=seed)
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(cfg)
for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)

print("段 ids:", [(s["id"], round(float(s["gvar"].mean()), 3)) for s in model.segments])
wi = names.index("wave")
Xa, Ya = np.array(test_sets[wi][0]), np.array(test_sets[wi][1])
n = len(Xa)

# ctx[#2] 低的输入长什么样
low = []
for i in range(n):
    x = Xa[i]
    p = model.perceive(x)
    ctx = dict(model._context(p))
    if ctx.get(2, 0.0) < 0.15:
        low.append((i, ctx, float(np.sum(x * x))))
print(f"\nctx[#2]<0.15 的输入 {len(low)} 个:")
for i, ctx, sq in low[:12]:
    top = sorted(ctx.items(), key=lambda kv: -kv[1])[:3]
    print(f"  输入{i}: ||x||²={sq:.2f}  ctx top3={top}")

# 错误样本构成
print("\n错误样本:")
nbad = 0
for i in range(n):
    x = Xa[i]
    p = model.perceive(x)
    pred, conf, selected, dists, ens, reflex, ctx = model.decide(p, count_energy=False)
    if pred != Ya[i]:
        nbad += 1
        if nbad <= 12:
            rels = np.array([c.cond_reliability(ctx) for c in model.cells])
            w = np.exp(cfg.ensemble_beta * rels[list(selected)])
            w = (w / w.sum()).round(2)
            ids = [model.cells[i].cell_id for i in selected]
            print(f"  输入{i}: y={Ya[i]} pred={pred} reflex={int(reflex)} ctx={ {k: round(v,2) for k,v in sorted(ctx.items(), key=lambda kv:-kv[1])[:3]} }")
            print(f"         选中{ids} w={list(w)}  各细胞意见={[int(np.argmax(d)) for d in [dists[i] for i in selected]]}")
