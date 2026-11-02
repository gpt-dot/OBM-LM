"""diag_scores.py —— 打印每个任务测试输入在各段上的上下文得分与读者结构。

回答：为什么所有输入都塌向段 #5（blobs3）？
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))
for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)

print("封存段:", [(s["id"], f"n={s['n']}", sorted(s['means'].keys())) for s in model.segments])
print("当前段:", model.seg_counter)

# 对每个任务取前 3 个测试输入，打印各段得分
for ti, (X, y) in enumerate(test_sets):
    print(f"\n===== 任务 {names[ti]} =====")
    for xi in range(3):
        p = model.perceive(X[xi])
        dists = [c.predict(p) for c in model.cells]
        # 复刻 _context 的内部计算以便打印
        cand = [(s["id"], s["oprofile"]) for s in model.segments]
        if model.cur_op:
            cand.append((model.seg_counter, model._cur_profile()))
        n_seg = len(cand)
        raw = [[None] * n_seg for _ in model.cells]
        wts = [[0.0] * n_seg for _ in model.cells]
        for i, c in enumerate(model.cells):
            d = model._sharp(dists[i], model.cfg.op_temp)
            dn = float(np.linalg.norm(d))
            if dn < 1e-9:
                continue
            for si, (seg_id, oprofile) in enumerate(cand):
                row = oprofile.get(c.cell_id)
                if not row:
                    continue
                raw[i][si] = max(float(np.dot(model._sharp(pr, model.cfg.op_temp), d))
                                 for pr in row.values())
                rel = c.seg_rel.get(seg_id)
                wts[i][si] = (0.25 if rel is None else max(0.0, (rel - 0.5) * 2.0))
        scores = []
        for si in range(n_seg):
            num, den = 0.0, 0.0
            for i in range(len(model.cells)):
                if raw[i][si] is None or wts[i][si] <= 0.0:
                    continue
                vals = [v for v in raw[i] if v is not None]
                centered = raw[i][si] - float(np.mean(vals))
                num += wts[i][si] * centered
                den += wts[i][si]
            scores.append(num / den if den > 1e-9 else None)
        ctx = model._context(p, dists)
        line = f"  输入{xi} (y={y[xi]}): "
        for si in range(n_seg):
            sid = cand[si][0]
            readers = [(model.cells[i].cell_id, round(wts[i][si], 2),
                        None if raw[i][si] is None else round(raw[i][si], 3))
                       for i in range(len(model.cells)) if wts[i][si] > 0]
            sc = scores[si]
            line += f"\n      段#{sid}: score={None if sc is None else round(sc, 4)} "
            line += f"ctx={ctx.get(sid, 0):.3f} 读者(cell,w,raw)={readers}"
        print(line)
