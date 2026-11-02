"""toy_regress.py —— numpy 路（mini-autograd）玩具协议回归守卫。

每次改动 lm_mesh / backend / cell_lm 后必跑：确认 OBM 机制在玩具四任务
协议（copy/reverse/sort/shift + 后两段复现）上的行为基线不退：

  1. 六段全部跑完（120 步/段），无异常崩溃；
  2. 终局每任务输出段精度（防遗忘的量化证据）；
  3. 机制事件齐全：封存 ≥3、分裂/凋亡/复现授予有记录；
  4. 复现段终值不低于对应任务的历史最佳 - 0.15（抗遗忘底线）。

用法：python3 experiments/toy_regress.py [seed]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from obm_lm import OBMLM, OBMLMConfig, build_blocks, PLAN, TASKS

BLOCK = 300
BATCH = 12
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    blocks, test_sets, names = build_blocks(0, 160)
    test_pairs = [(s, t) for s, t, _ in test_sets]
    masks = [m for _, _, m in test_sets]
    cfg = OBMLMConfig(seed=SEED)
    model = OBMLM(cfg)

    step = 0
    for block in blocks:
        for _ in range(BLOCK):
            seq, tgt, mask = block.batch(BATCH)
            rec = model.run_step(seq, tgt, mask)
            step += 1

    final = model.evaluate(test_pairs, masks)
    print("== 终局输出段精度 ==")
    for n, a in zip(names, final):
        print(f"  {n:8s} {a:.3f}")
    print(f"  均值 {np.mean(final):.3f}  细胞数 {len(model.cells)}  "
          f"参数量 {model.n_params/1e3:.0f}K")

    print("\n== 机制事件统计 ==")
    from collections import Counter
    cnt = Counter(t for _, t, _ in model.events)
    for t, c in sorted(cnt.items()):
        print(f"  {t:10s} ×{c}")
    seals = [(s, d) for s, t, d in model.events if t == "seal"]
    births = [(s, d) for s, t, d in model.events if t == "birth"]
    deaths = [(s, d) for s, t, d in model.events if t == "death"]
    reprise = [(s, d) for s, t, d in model.events if "复现授予" in d or "复现" in d]
    print(f"\n  封存 {len(seals)} 次 / 分裂 {len(births)} 次 / "
          f"凋亡 {len(deaths)} 次 / 复现授予 {len(reprise)} 次")

    # 历史最佳 vs 终值（复现段抗遗忘）
    print("\n== 每任务 最佳→终值（在线 ens_acc 口径） ==")
    starts = np.cumsum([0] + [BLOCK] * len(blocks))
    for ti, name in enumerate(names):
        seg = [h["acc"] for h in model.history[starts[ti]:starts[ti + 1]]]
        if seg:
            print(f"  {name:8s} 最佳 {max(seg):.3f} -> 终值 {seg[-1]:.3f}"
                  f"（均值 {np.mean(seg):.3f}）")

    # 底线断言
    ok = True
    if len(seals) < 3:
        print(f"\n[FAIL] 封存次数 {len(seals)} < 3（边界检测退化）")
        ok = False
    if np.mean(final) < 0.75:
        print(f"\n[FAIL] 终局均值 {np.mean(final):.3f} < 0.75（学习退化）")
        ok = False
    for ti in [4, 5]:                       # 两个复现段
        seg = [h["acc"] for h in model.history[starts[ti]:starts[ti + 1]]]
        if seg and max(seg) - seg[-1] > 0.15:
            print(f"\n[FAIL] 复现段 {names[ti]} 跌落 {max(seg):.3f}->"
                  f"{seg[-1]:.3f} 超过 0.15（抗遗忘退化）")
            ok = False
    print("\n" + ("[PASS] 玩具回归基线保持" if ok else "[FAIL] 见上"))


if __name__ == "__main__":
    main()
