"""smoke.py —— 意见指纹上下文推理的种子 1 冒烟测试。

精确复测上次失败的指标：
  * 每个任务测试输入的 top-1 段归属（上次 linear 被 wave 片段 #2 抢走：
    Counter({2: 18, 0: 12, ...})）
  * 各任务最终测试精度（目标：linear 从 0.690 升到 >= 0.85）
  * 封存/合并事件是否干净
  * 细胞的条件合同状态（专才是否存活）
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

PLAN = ["linear", "xor", "wave", "blobs3", "linear", "xor", "moons"]
BLOCK = 600
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    train_blocks, test_sets, names = build_stream()
    model = OuroborosMesh(OBMConfig(seed=SEED))

    step = 0
    for bi, (X, y) in enumerate(train_blocks):
        for x, yy in zip(X, y):
            model.run_step(x, yy)
            step += 1

    # ---------- 1. 各任务最终精度 ----------
    final = model.evaluate(test_sets)
    print("== 最终测试精度 ==")
    for n, a in zip(names, final):
        print(f"  {n:8s} {a:.3f}")
    print(f"  均值     {np.mean(final):.3f}")

    # ---------- 2. 事件流 ----------
    print("\n== 事件 ==")
    for st, typ, detail in model.events:
        print(f"  [{st:5d}] {typ:6s} {detail}")

    # ---------- 3. 段 -> 任务 归属（从事件推导） ----------
    # 每个封存事件按"保留数据的中点"归入某个 block：封存检测有滞后，
    # 且缓冲只留最近 400 条、封存时再去掉最旧 35%，保留跨度约
    # [st-260, st]，中点 st-130 所在的 block 才是这段数据的真主。
    seg_task = {}          # seg_id -> task name
    task_segs = {}         # task name -> set(seg_id)
    for st, typ, detail in model.events:
        if typ != "seal":
            continue
        blk = min(max((st - 130) // BLOCK, 0), len(PLAN) - 1)
        task = PLAN[blk]
        if "任务复现" in detail and "概念段 #" in detail:
            # "任务复现：N 个片段并入概念段 #M（库存 K）"
            import re
            m = re.search(r"概念段 #(\d+)", detail)
            old = int(m.group(1))
            # 新段号退役：找到刚退役的号（当前任务此前可能占用的段）
            for s, t in list(seg_task.items()):
                if t == task and s != old:
                    task_segs[task].discard(s)
                    del seg_task[s]
            seg_task[old] = task
            task_segs.setdefault(task, set()).add(old)
        else:
            import re
            m = re.search(r"概念段 #(\d+) 封存", detail)
            new_id = int(m.group(1))
            seg_task[new_id] = task
            task_segs.setdefault(task, set()).add(new_id)
    print("\n== 段 -> 任务（从事件推导） ==")
    for s in sorted(seg_task):
        print(f"  段 #{s}: {seg_task[s]}")
    for t, ss in task_segs.items():
        print(f"  任务 {t:8s} -> 段 {sorted(ss)}")

    # ---------- 4. 领地归属诊断（核心指标） ----------
    print("\n== 测试输入的 top-1 段归属 ==")
    for ti, (X, y) in enumerate(test_sets):
        tname = names[ti]
        want = task_segs.get(tname, set())
        cnt = Counter()
        hit = 0
        for x in X:
            p = model.perceive(x)
            ctx = model._context(p)
            top = max(ctx.items(), key=lambda kv: kv[1])[0]
            cnt[top] += 1
            hit += int(top in want)
        print(f"  {tname:8s} (期望段 {sorted(want)}): 命中 {hit}/{len(X)}"
              f"  top1分布 {dict(sorted(cnt.items()))}")

    # ---------- 5. 细胞状态 ----------
    print("\n== 细胞条件合同 ==")
    for c in model.cells:
        rels = {s: round(v, 3) for s, v in sorted(c.seg_rel.items())}
        print(f"  Cell#{c.cell_id}: seg_rel={rels} niche={c.niche():.3f} "
              f"act={c.activations} ok={c.correct}")


if __name__ == "__main__":
    main()
