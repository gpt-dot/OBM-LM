"""diag_attr.py —— 为什么 linear 测试输入归属到段 #8（xor-2）？

按真实 PLAN 边界（每块 600 步）核对段→任务，再对 linear 测试输入
逐细胞打印：锐化意见、各段原始余弦、极差、信任权重、中心化贡献。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PLAN = ["linear", "xor", "wave", "blobs3", "linear", "xor", "moons"]
BLOCK = 600

train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))
step = 0
for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)
        step += 1

print("总步数:", step)
print("\n== 段 -> 任务（按封印步归属真实 PLAN 块；边界检测有滞后，"
      "封印步落在哪块就是哪块） ==")
for e in model.events:
    if e[1] == "seal":
        st = e[0]
        blk = min(st // BLOCK, len(PLAN) - 1)
        print(f"  [{st:5d}] {e[2][:44]:44s} -> 块{blk} = {PLAN[blk]}")
print("  当前段 id:", model.seg_counter, "（块", min(step // BLOCK, 6), "=",
      PLAN[min(step // BLOCK, 6)], "）")

print("\n== 细胞条件合同（seg_rel） ==")
for c in model.cells:
    rel = {k: round(v, 2) for k, v in sorted(c.seg_rel.items())}
    print(f"  Cell#{c.cell_id}: niche={c.niche():.2f} act={c.activations} "
          f"ok={c.correct} ({c.correct/max(c.activations,1):.2f})")
    print(f"    seg_rel={rel}")

TASK = "linear"
ti = names.index(TASK)
X, y = test_sets[ti]
print(f"\n== 任务 {TASK} 前 3 个测试输入的逐段得分 ==")
for xi in range(3):
    p = model.perceive(X[xi])
    dists = [c.predict(p) for c in model.cells]
    cand = [(s["id"], s["oprofile"]) for s in model.segments]
    if model.cur_op:
        cand.append((model.seg_counter, model._cur_profile()))
    n_seg = len(cand)
    raw = [[None] * n_seg for _ in model.cells]
    wts = [[0.0] * n_seg for _ in model.cells]
    for i, c in enumerate(model.cells):
        d = model._sharp(dists[i], model.cfg.op_temp)
        if np.linalg.norm(d) < 1e-9:
            continue
        for si, (seg_id, oprofile) in enumerate(cand):
            row = oprofile.get(c.cell_id)
            if not row:
                continue
            raw[i][si] = max(float(np.dot(model._sharp(pr, model.cfg.op_temp), d))
                             for pr in row.values())
            rel = c.seg_rel.get(seg_id)
            wts[i][si] = (0.25 if rel is None
                          else max(0.0, (rel - 0.5) * 2.0))
    spread = []
    for i in range(len(model.cells)):
        vals = [v for v in raw[i] if v is not None]
        spread.append(max(vals) - min(vals) if len(vals) >= 2 else 0.0)
    scores = []
    for si in range(n_seg):
        num, den = 0.0, 0.0
        for i in range(len(model.cells)):
            if raw[i][si] is None or wts[i][si] <= 0 or spread[i] <= 0:
                continue
            vals = [v for v in raw[i] if v is not None]
            centered = raw[i][si] - float(np.mean(vals))
            num += wts[i][si] * spread[i] * centered
            den += wts[i][si] * spread[i]
        scores.append(num / den if den > 1e-9 else 0.0)
    ctx = model._context(p, dists)
    print(f"\n  输入{xi} y={y[xi]}  意见={np.round(dists[0], 2)}")
    for si in range(n_seg):
        sid = cand[si][0]
        readers = [(model.cells[i].cell_id, round(wts[i][si], 2),
                    round(spread[i], 2),
                    None if raw[i][si] is None else round(raw[i][si], 3))
                   for i in range(len(model.cells)) if raw[i][si] is not None]
        print(f"    段#{sid}: score={scores[si]:+.4f} ctx={ctx.get(sid,0):.3f} "
              f"读者(cell,w,spread,raw)={readers}")
