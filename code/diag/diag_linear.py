"""linear 测试输入的 ctx/选中集/权重诊断（对比 sharp 前后）。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
uni = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
cfg = OBMConfig(seed=seed, ctx_temp=temp, ctx_uniform=uni)
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(cfg)
for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)

print("段 ids:", [s["id"] for s in model.segments], " 当前段:", model.seg_counter)
li = names.index("linear")
Xa, Ya = np.array(test_sets[li][0]), np.array(test_sets[li][1])
n = len(Xa)

# 每细胞 linear 单独精度
for idx, c in enumerate(model.cells):
    ok = sum(int(np.argmax(c.predict(model.perceive(x))) == yy) for x, yy in zip(Xa, Ya))
    print(f"  索引{idx} id#{c.cell_id}: linear {ok/n:.3f}  (seg_rel: { {k: round(v,2) for k,v in c.seg_rel.items()} })")

# ctx 与选中集统计
want_seg = [s["id"] for s in model.segments if abs(s["gmean"][0]) < 1e9][:1]
ctx_top = {}
sel_ids = []
acc_ok = 0
ctx0 = []
for i in range(n):
    x = Xa[i]
    p = model.perceive(x)
    pred, conf, selected, dists, ens, reflex, ctx = model.decide(p, count_energy=False)
    acc_ok += int(pred == Ya[i])
    top = max(ctx.items(), key=lambda kv: kv[1])[0]
    ctx_top[top] = ctx_top.get(top, 0) + 1
    ctx0.append(dict(ctx).get(0, 0.0))
    sel_ids.append(tuple(sorted(model.cells[i].cell_id for i in selected)))
print(f"\nlinear 集成精度: {acc_ok/n:.3f}")
print(f"ctx top-1 段分布: {ctx_top}")
print(f"ctx[#0] mean={np.mean(ctx0):.3f}")
from collections import Counter
print("选中集(细胞id) top5:", Counter(sel_ids).most_common(5))
