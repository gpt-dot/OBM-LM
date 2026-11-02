"""diag_dream600.py —— 插桩诊断 step 600 的梦境：谁被选中、对错如何。

假设：专才在梦里反刍"高意外度"样本时大量翻车，信誉被 dream_alpha 拖垮。
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))

# 拦截 _rehearse，记录每次 consolidate 的细节
orig_consolidate = None
log = []

step = 0
for bi, (X, y) in enumerate(train_blocks):
    for x, yy in zip(X, y):
        model.run_step(x, yy)
        step += 1
        if step > 620:
            break
    if step > 620:
        break

print("600 步后细胞状态：")
for c in model.cells:
    print(f"  {c!r} seg_rel={ {k: round(v,3) for k,v in sorted(c.seg_rel.items())} }")

# ---- 手动重放到 600 步并插桩梦境 ----
model2 = OuroborosMesh(OBMConfig(seed=SEED))

# 包住 _rehearse：统计每个细胞在梦里的选中次数/对错
stats = {}
orig_rehearse = model2._rehearse

def rehearsed(pools):
    cfg = model2.cfg
    rng = model2.rng
    for _ in range(cfg.dream_batches):
        batch = []
        for _ in range(cfg.dream_batch):
            seg_id, samples = pools[rng.integers(len(pools))]
            p, y = samples[rng.integers(len(samples))]
            batch.append((seg_id, p, y))

        def consolidate(seg_id, p, y):
            ctx = {seg_id: 1.0}
            dists = [c.predict(p) for c in model2.cells]
            rels = np.array([c.cond_reliability(ctx) for c in model2.cells])
            selected, _ = model2._select(dists, rels)
            for i in selected:
                c = model2.cells[i]
                ok = int(np.argmax(dists[i]) == y)
                st = stats.setdefault(c.cell_id, [0, 0])
                st[0] += 1
                st[1] += ok
                c.learn(p, y, cfg.dream_lr if ok else cfg.lr_breach)
                c.note(seg_id, ok, cfg.dream_alpha)
                c.seg_note[seg_id] = model2.step

        for seg_id, p, y in batch:
            consolidate(seg_id, p, y)
        if cfg.counterfactual:
            hard = sorted(batch, key=lambda t: -model2._cell_loss(t[1], t[2]))
            for seg_id, p, y in hard[:max(1, len(batch) // 4)]:
                for _ in range(2):
                    consolidate(seg_id, p + cfg.cf_noise * rng.normal(size=p.shape), y)

model2._rehearse = rehearsed

step = 0
for bi, (X, y) in enumerate(train_blocks):
    for x, yy in zip(X, y):
        model2.run_step(x, yy)
        step += 1
    if step >= 600:
        break

print("\n插桩运行到 600 步（含 200/400/600 三次梦境）后：")
for c in model2.cells:
    print(f"  {c!r} seg_rel={ {k: round(v,3) for k,v in sorted(c.seg_rel.items())} }")
print("\n梦境中的选中/答对统计（cell_id: [选中次数, 答对次数]）：")
for cid, (n, ok) in sorted(stats.items()):
    print(f"  Cell#{cid}: {n} 次选中, {ok} 次答对 ({ok/max(n,1):.0%})")

# 各细胞在 block0 训练集上的真实精度（对照：梦里翻车是不是因为样本特别难）
X0, y0 = train_blocks[0]
print("\n各细胞在 block0（linear）全部 600 个训练样本上的精度：")
for c in model2.cells:
    acc = np.mean([int(np.argmax(c.predict(model2.perceive(x))) == yy) for x, yy in zip(X0, y0)])
    print(f"  Cell#{c.cell_id}: {acc:.3f}")
