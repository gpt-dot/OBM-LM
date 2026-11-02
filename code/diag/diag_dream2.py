"""diag_dream2.py —— 精确测量：梦境样本分布的难度分解。

对每个细胞分别测量在以下集合上的精度：
  1. block0 全量 600 训练样本
  2. 梦境实际抽样的缓冲（最后 400 条）
  3. 高意外度 top-25%（按 surprise 排序）
  4. 反事实噪声样本（hard top-1/4 + 0.25*N(0,1)，两次）
并统计梦境 consolidate 的调用次数（校验诊断自身的一致性）。
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))

n_calls = [0]
orig_rehearse = model._rehearse

def rehearsed(pools):
    cfg = model.cfg
    rng = model.rng
    for _ in range(cfg.dream_batches):
        batch = []
        for _ in range(cfg.dream_batch):
            seg_id, samples = pools[rng.integers(len(pools))]
            p, y = samples[rng.integers(len(samples))]
            batch.append((seg_id, p, y))

        def consolidate(seg_id, p, y):
            n_calls[0] += 1
            ctx = {seg_id: 1.0}
            dists = [c.predict(p) for c in model.cells]
            rels = np.array([c.cond_reliability(ctx) for c in model.cells])
            selected, _ = model._select(dists, rels)
            for i in selected:
                c = model.cells[i]
                ok = int(np.argmax(dists[i]) == y)
                c.learn(p, y, cfg.dream_lr if ok else cfg.lr_breach)
                c.note(seg_id, ok, cfg.dream_alpha)
                c.seg_note[seg_id] = model.step

        for seg_id, p, y in batch:
            consolidate(seg_id, p, y)
        if cfg.counterfactual:
            hard = sorted(batch, key=lambda t: -model._cell_loss(t[1], t[2]))
            for seg_id, p, y in hard[:max(1, len(batch) // 4)]:
                for _ in range(2):
                    consolidate(seg_id, p + cfg.cf_noise * rng.normal(size=p.shape), y)

model._rehearse = rehearsed

step = 0
for bi, (X, y) in enumerate(train_blocks):
    for x, yy in zip(X, y):
        model.run_step(x, yy)
        step += 1
    if step >= 600:
        break

print(f"consolidate 总调用次数: {n_calls[0]}（预期 3 梦 × (8×32 + 8×2) = 816）")
print("\n600 步后细胞状态：")
for c in model.cells:
    print(f"  {c!r} seg_rel={ {k: round(v,3) for k,v in sorted(c.seg_rel.items())} }")

X0, y0 = train_blocks[0]

def cell_acc(c, samples):
    if not samples:
        return float("nan")
    # 注意：缓冲里存的已经是脉冲 p，不能再过 perceive
    return float(np.mean([int(np.argmax(c.predict(p)) == yy)
                          for p, yy in samples]))

# 缓冲快照：重放到 600 步，取当前 buffer
buf = [(p, y) for p, y, s, st in model.buffer]
print(f"\n缓冲样本数: {len(buf)}")
surps = sorted(buf, key=lambda t: -0)  # 占位
# 按意外度排序需要 surprise，重新从 buffer 四元组取
full_buf = list(model.buffer)
by_surp = sorted(full_buf, key=lambda t: -t[2])
hard25 = [(p, y) for p, y, s, st in by_surp[:max(1, len(by_surp) // 4)]]
print(f"高意外度 top25%: {len(hard25)} 条")

rng = np.random.default_rng(999)
noisy = [(p + model.cfg.cf_noise * rng.normal(size=p.shape), y) for p, y in hard25]

sets = {
    "全量600": [(model.perceive(x), yy) for x, yy in zip(X0, y0)],
    "缓冲400": buf,
    "高意外25%": hard25,
    "反事实噪声": noisy,
}
print("\n各细胞在不同样本集上的精度：")
hdr = f"{'cell':8s}" + "".join(f"{k:>10s}" for k in sets)
print(hdr)
for c in model.cells:
    row = f"C{c.cell_id:<7d}"
    for k, ss in sets.items():
        row += f"{cell_acc(c, ss):>10.3f}"
    print(row)
print("\n参照：600 步后的集成（ensemble）在同一些集合上的精度：")
for k, ss in sets.items():
    acc = np.mean([int(model.decide(p, count_energy=False)[0] == yy)
                   for p, yy in ss])
    print(f"  {k:10s}: {acc:.3f}")
