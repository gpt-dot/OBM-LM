"""封存时把 buffer 内容与真实输入流逐样本比对。"""
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

seed = 1
cfg = OBMConfig(seed=seed)
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(cfg)

# 重建真实输入流（按 step 顺序），用于比对
true_X = np.vstack([np.array(X) for X, _ in train_blocks])
true_task = np.concatenate([[i] * len(X) for i, (X, _) in enumerate(train_blocks)])
print("train_blocks 各块长度:", [len(X) for X, _ in train_blocks])

orig_seal = model._seal_segment
def spy_seal():
    seg_id = model.seg_counter
    if seg_id in (0, 2, 4) and len(model.buffer) >= 20:
        entries = sorted(model.buffer, key=lambda t: t[3])
        cut = int(len(entries) * cfg.buf_trim)
        kept = entries[cut:]
        # buffer 里每个样本的 step -> 真实任务
        steps = np.array([t[3] for t in kept])
        tasks_of = true_task[steps] if steps.max() < len(true_task) else None
        xs = np.array([model._decode(p) for p, y, *_ in kept])
        print(f"\n[段#{seg_id} 封存] 修剪后 {len(kept)} 个, step=[{steps.min()},{steps.max()}]")
        if tasks_of is not None:
            import collections
            print(f"  buffer 样本的真实任务组成: {dict(collections.Counter(tasks_of))}")
            # 与真实输入直接比对 decode 误差
            real = true_X[steps]
            err = np.mean(np.linalg.norm(xs - real, axis=1))
            print(f"  decode 与真实输入的平均误差: {err:.4f}")
            # 分任务算 var
            for ti in sorted(set(tasks_of)):
                m = tasks_of == ti
                print(f"    任务{names[ti]}({m.sum()}个): decode var mean={xs[m].var(0).mean():.3f} 真实 var mean={real[m].var(0).mean():.3f}")
    return orig_seal()
model._seal_segment = spy_seal

for X, y in train_blocks:
    for x, yy in zip(X, y):
        model.run_step(x, yy)
