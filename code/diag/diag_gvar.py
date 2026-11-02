"""追踪段 #2 封存时的 buffer 组成与 gvar。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cfg = OBMConfig(seed=seed)
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(cfg)

# 猴补丁：封存时打印 buffer 统计
orig_seal = model._seal_segment
def spy_seal():
    if len(model.buffer) >= 20:
        entries = sorted(model.buffer, key=lambda t: t[3])
        cut = int(len(entries) * cfg.buf_trim)
        kept = entries[cut:]
        xs = np.array([model._decode(p) for p, y, *_ in kept])
        steps = [t[3] for t in kept]
        ys = np.array([t[1] for t in kept])
        print(f"[封存 段#{model.seg_counter}] buffer={len(model.buffer)} 修剪后={len(kept)} "
              f"step范围=[{min(steps)},{max(steps)}]")
        print(f"   decode var mean={xs.var(0).mean():.3f} 前两维=[{xs.var(0)[0]:.3f},{xs.var(0)[1]:.3f}] "
              f"类别分布={np.bincount(ys, minlength=3)}")
        # 与整个当前段（从上一封存到现在）的真实输入对比
        print(f"   -> gvar = {list(np.round(xs.var(0), 3))}")
    return orig_seal()
model._seal_segment = spy_seal

for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)

print("\n最终各段 gvar:")
for s in model.segments:
    print(f"  段#{s['id']}: gvar mean={s['gvar'].mean():.3f} 前两维=[{s['gvar'][0]:.3f},{s['gvar'][1]:.3f}] n={s['n']}")
