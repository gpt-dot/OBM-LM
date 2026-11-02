"""diag_norm.py —— 各任务的脉冲范数分布 + 封存点的 recent/older 值。"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))

PLAN = ["linear", "xor", "wave", "blobs3", "linear", "xor", "moons"]
# 1) 各任务脉冲范数
print("== 各任务 ||p|| 与 ||x|| ==")
for (X, y), name in zip(train_blocks, PLAN):
    P = np.array([model.perceive(x) for x in X[:200]])
    Xn = np.linalg.norm(X[:200], axis=1)
    print(f"  {name:8s} ||x||={Xn.mean():.2f}±{Xn.std():.2f}  "
          f"||p||={np.linalg.norm(P, axis=1).mean():.2f}±"
          f"{np.linalg.norm(P, axis=1).std():.2f}")

# 2) 封存点的 recent/older
orig = model._seal_segment
def spy():
    if len(model.buffer) >= 120:
        recent = np.mean([np.linalg.norm(e[0]) for e in model.buffer[-60:]])
        older = np.mean([np.linalg.norm(e[0]) for e in model.buffer[:-60]])
        print(f"  [seal@{model.step}] recent={recent:.3f} older={older:.3f} "
              f"diff={abs(recent-older)/older*100:.1f}% nbuf={len(model.buffer)}")
    return orig()
model._seal_segment = spy

step = 0
for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)
        step += 1
