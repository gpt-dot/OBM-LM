"""diag_acc.py —— 测量梦境学习对细胞真实精度的影响。

在每个梦境前后，测量所有细胞在 block0 全量训练集上的真实精度。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))
X0, y0 = train_blocks[0]
P0 = [model.perceive(x) for x in X0]


def accs(tag):
    row = f"[{model.step:5d}] {tag:10s}"
    for c in model.cells:
        a = float(np.mean([int(np.argmax(c.predict(p)) == yy) for p, yy in zip(P0, y0)]))
        row += f"  C{c.cell_id}={a:.3f}"
    print(row)


step = 0
dream_pts = {200, 400, 600}
for bi, (X, y) in enumerate(train_blocks):
    for x, yy in zip(X, y):
        pre = model.step in dream_pts
        if pre:
            accs("梦前")
        model.run_step(x, yy)
        if pre:
            accs("梦后")
        step += 1
    if step >= 620:
        break

print("\n结论：若梦后精度大幅低于梦前 -> 梦境学习在破坏细胞")
