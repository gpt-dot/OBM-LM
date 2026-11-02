"""diag_cells.py —— 每个细胞在每个任务测试集上的真实精度 + 意见锐度。

回答：1) 各细胞到底会什么（路由的天花板）？
      2) 意见指纹为什么没有区分度（意见到处都尖）？
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))
for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)

print("== 各细胞在各任务测试集上的精度 ==")
hdr = "  cell  " + "".join(f"{n:>8s}" for n in names) + "   平均"
print(hdr)
for c in model.cells:
    row = []
    for X, y in test_sets:
        ok = sum(int(np.argmax(c.predict(model.perceive(x))) == yy)
                 for x, yy in zip(X, y))
        row.append(ok / len(y))
    print(f"  #{c.cell_id:<4d} " + "".join(f"{v:8.2f}" for v in row)
          + f" {np.mean(row):7.2f}")

print("\n== 各细胞意见锐度（测试集上 sharp 后 top1 占比的均值） ==")
print("  cell  " + "".join(f"{n:>8s}" for n in names))
for c in model.cells:
    row = []
    for X, y in test_sets:
        sharp = []
        for x in X:
            d = model._sharp(c.predict(model.perceive(x)), model.cfg.op_temp)
            sharp.append(d.max())
        row.append(np.mean(sharp))
    print(f"  #{c.cell_id:<4d} " + "".join(f"{v:8.2f}" for v in row))

print("\n== 各段意见剖面里各细胞的行尖度（封存时平均） ==")
for s in model.segments:
    parts = []
    for cid, rows in s["oprofile"].items():
        v = np.mean([model._sharp(r, model.cfg.op_temp).max() for r in rows.values()])
        parts.append(f"#{cid}:{v:.2f}")
    print(f"  段#{s['id']} (n={s['n']}): " + " ".join(parts))
