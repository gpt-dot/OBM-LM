"""诊断：全程跟踪 copy 测试精度 + 每个细胞被训练在哪个任务上。

目的：定位 copy 从 ~1.0 崩到 0.761 的时间窗与机制。
  - 每 20 步评估 copy 测试集（前 48 条，4 个 batch）
  - 每步记录 run_step 选中的细胞、各细胞当步精度
  - 按 block 统计每个细胞实际接受训练的步数（负迁移检测）
  - 梦境前后对比 copy 专精细胞的精度
"""
import sys
import time

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # 位级可复现：固定 BLAS 线程（见 obm_lm/__init__.py）

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.lm_mesh import OBMLM, OBMLMConfig, _OUT_MASK
from obm_lm.langs import build_blocks, PLAN, TASKS

BLOCK = 600
BATCH = 12
N_EVAL = 48          # copy 测试集取前 48 条
EVAL_EVERY = 20


def main():
    import os
    seed = int(os.environ.get("DIAG_SEED", "1"))
    cfg = OBMLMConfig(seed=seed)
    # 复现授予的对照开关：DIAG_REPRISE_D=-1 时 _reprise_seg 永不返回段
    # （距离不可能 ≤ -1），完整复现补丁前的授予语义——用于同种子 A/B 对比
    rd = os.environ.get("DIAG_REPRISE_D")
    if rd is not None:
        cfg.award_reprise_d = float(rd)
        print(f"[cfg] award_reprise_d={cfg.award_reprise_d}（对照运行）")
    model = OBMLM(cfg)
    blocks, test_sets, names = build_blocks(0, 160)
    copy_ti = TASKS.index("copy")
    cseq, ctgt, cmask = test_sets[copy_ti]
    cseq, ctgt = cseq[:N_EVAL], ctgt[:N_EVAL]

    def copy_acc():
        ok = tot = 0
        for b in range(0, N_EVAL, BATCH):
            chunk, t = cseq[b:b + BATCH], ctgt[b:b + BATCH]
            pred, _ = model._eval_batch(chunk, t, cmask)
            ok += int(np.sum(pred[:, cmask] == t[:, cmask]))
            tot += int(cmask.sum()) * len(chunk)
        return ok / max(tot, 1)

    # 每个细胞的训练台账：cell_id -> {task: steps}
    train_ledger = {c.cell_id: {k: 0 for k in TASKS} for c in model.cells}
    sel_trace = []          # (step, selected, cell_accs)
    dream_marks = []        # (step, copy_acc before/after)
    ev_printed = 0
    t0 = time.time()
    step = 0
    total = len(blocks) * BLOCK
    for bi, block in enumerate(blocks):
        task = block.name
        for _ in range(BLOCK):
            seq, tgt, mask = block.batch(BATCH)
            d = model.run_step(seq, tgt, mask)
            sel_trace.append((step, list(d["selected"]),
                              [round(a, 2) for a in d["cell_accs"]]))
            for i in d["selected"]:
                cid = model.cells[i].cell_id
                if cid not in train_ledger:
                    train_ledger[cid] = {k: 0 for k in TASKS}
                train_ledger[cid][task] += 1
            if step % EVAL_EVERY == 0:
                a = copy_acc()
                print(f"  step {step:4d} [{task:7s}] copy={a:.3f} "
                      f"cells={len(model.cells)} [{time.time()-t0:.0f}s]",
                      flush=True)
            # 事件
            for e in model.events[ev_printed:]:
                print(f"    [event] {e[0]:5d} {e[1]}: {e[2]}", flush=True)
            ev_printed = len(model.events)
            # 复现授予示踪：award 带 match 键 = 本步走了复现授予分支
            # （每 50 步桶至多打一行，避免刷屏）
            if model.award is not None and model.award.get("match") is not None:
                key = (model.award["match"], step // 50)
                if getattr(main, "_last_reprise", None) != key:
                    main._last_reprise = key
                    print(f"    [reprise] step {step}: 授予段#{key[0]} 专才 "
                          f"cell#{model.award['cid']}", flush=True)
            step += 1

    print("\n=== copy 精度崩塌点前后（每步步长）===")
    accs = []
    step = 0
    # 重算不可行（模型已变），改从 sel_trace + history 间接看；
    # 直接输出末态详情
    print("final copy:", round(copy_acc(), 3))
    print("\n=== 训练台账（每细胞在各任务上的受训步数）===")
    for cid, led in sorted(train_ledger.items()):
        print(f"  cell #{cid}: {led}")
    print("\n=== 各细胞最终账本 ===")
    for c in model.cells:
        print(f"  cell #{c.cell_id}: seg_rel="
              f"{ {k: round(v, 2) for k, v in c.seg_rel.items()} } "
              f"seg_best={ {k: round(v, 2) for k, v in c.seg_best.items()} }")
    print("\n=== copy 段(#0)库存样本数 ===",
          [len(s["samples"]) for s in model.segments])
    # 评估期选路细节
    for ti, row in enumerate(model.eval_detail(
            [(s, t) for s, t, _ in test_sets],
            [m for _, _, m in test_sets])):
        print(f"  [eval] {names[ti]:7s} ctx_top=段#{row['ctx_top']} "
              f"{row['ctx_w']} sel={row['sel']} w={row['w']} "
              f"rels={row['rels']} cell_accs={row['cell_accs']}")
    # 每个细胞在 copy 测试集上的裸精度
    print("\n=== 各细胞在 copy 测试集上的裸精度 ===")
    for c in model.cells:
        ok = tot = 0
        for b in range(0, N_EVAL, BATCH):
            chunk, t = cseq[b:b + BATCH], ctgt[b:b + BATCH]
            pred = np.argmax(c.forward(chunk).data, -1)
            ok += int(np.sum(pred[:, cmask] == t[:, cmask]))
            tot += int(cmask.sum()) * len(chunk)
        print(f"  cell #{c.cell_id}: {ok / max(tot, 1):.3f}")


if __name__ == "__main__":
    main()
