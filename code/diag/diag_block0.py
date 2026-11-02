"""diag_block0.py —— 诊断：block 0（linear）期间细胞的可靠性轨迹与曝光。

回答：linear 专才是"从未学好"还是"学好后被凋谢/ decay 杀死"？
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))

step = 0
win_correct = []
sel_counts = {}
for bi, (X, y) in enumerate(train_blocks):
    blk_start = step
    for x, yy in zip(X, y):
        rec = model.run_step(x, yy)
        win_correct.append(rec["correct"])
        step += 1
        if step % 100 == 0:
            w = win_correct[-100:]
            cells = " | ".join(
                f"C{c.cell_id}:r0={c.seg_rel.get(0, float('nan')):.2f}"
                f",r{bi}={c.seg_rel.get(model.seg_counter, float('nan')):.2f}"
                for c in model.cells)
            print(f"[{step:5d}] blk{bi}({names[bi]:7s}) acc100={np.mean(w):.3f} "
                  f"seg={model.seg_counter} ncells={len(model.cells)} {cells}")
        if step >= 1400:
            break
    if step >= 1400:
        break
    # 段边界处打印封存详情
    print(f"----- block {bi} ({names[bi]}) 结束于 step {step}，seg_counter={model.seg_counter}")

print("\n== 1400 步时的细胞状态 ==")
for c in model.cells:
    print(f"  {c!r}  seg_rel={ {k: round(v,3) for k,v in sorted(c.seg_rel.items())} }")
