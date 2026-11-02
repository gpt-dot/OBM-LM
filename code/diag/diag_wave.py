"""diag_wave.py —— wave 测试输入：ctx/rels/选中/对错 全链路。"""
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

ti = names.index("wave")
X, y = test_sets[ti]
print("段:", [(s["id"], np.round(s["gvar"], 2).tolist()) for s in model.segments])
print("当前段:", model.seg_counter, "cur_x var:", np.round(
    np.maximum(model.cur_x["sq"]/model.cur_x["n"]
               - (model.cur_x["sum"]/model.cur_x["n"])**2, 0.05), 2).tolist())
print()
ok_cnt = 0
for xi in range(12):
    p = model.perceive(X[xi])
    ctx = model._context(p)
    top = sorted(ctx.items(), key=lambda kv: -kv[1])[:3]
    dists = [c.predict(p) for c in model.cells]
    rels = np.array([c.cond_reliability(ctx) for c in model.cells])
    selected, reflex = model._select(dists, rels)
    w = np.exp(model.cfg.ensemble_beta * rels[selected]); w /= w.sum()
    ens = sum(wi * dists[i] for wi, i in zip(w, selected))
    pred = int(np.argmax(ens))
    ok = int(pred == y[xi])
    ok_cnt += ok
    rels_s = " ".join(f"#{c.cell_id}:{r:.2f}" for c, r in zip(model.cells, rels))
    print(f"  输入{xi} y={y[xi]} pred={pred} {'OK' if ok else 'X'} reflex={int(reflex)} "
          f"ctx={[(k, round(v,2)) for k,v in top]}")
    print(f"    rels: {rels_s}")
    print(f"    selected={[model.cells[i].cell_id for i in selected]} "
          f"w={np.round(w,2).tolist()}")
print(f"\n前12个 wave 输入对 {ok_cnt}/12")
