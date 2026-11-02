"""OBM 功能判定：加载规模 run 的 metrics.json，客观判定核心机制是否点火。

硬指标：
  1) 事件：seal（封存）/ birth（分裂）/ death（凋亡）/ reprise（复现授予）是否发生
  2) 合同：是否有 cell 的 seg_rel >= 0.55（条件合同市场真正形成分工）
  3) 效果：final ppl 与 best_ppl（抗遗忘量）
用法：python3 experiments/diag_events.py results_scale/fast_pilot2/metrics.json
"""
import json, sys

def verdict(m):
    ev = m.get("events_all", [])
    evk = {}
    for s, k, msg in ev:
        evk[k] = evk.get(k, 0) + 1
    print("=" * 60)
    print("OBM 功能判定  (run: %s)" % m.get("scale", "?"))
    print("=" * 60)
    print("[机制事件]")
    for name, key in [("封存 seal", "seal"), ("分裂 birth", "birth"),
                      ("凋亡 death", "death"), ("复现授予 reprise", "reprise")]:
        n = evk.get(key, 0)
        flag = "✓" if n > 0 else "✗ 未点火"
        print(f"  {name:16s}: {n:3d}  {flag}")
    print(f"  总事件数: {len(ev)}")
    print()
    print("[合同（分工）]  过线阈值 seg_rel >= 0.55")
    any_contract = False
    for c in m.get("territory", []):
        srs = c.get("seg_rel", {})
        over = [k for k, v in srs.items() if v >= 0.55]
        if over:
            any_contract = True
        print(f"  cell#{c['cell']}: " + ", ".join(
            f"{k}:{v:.3f}" for k, v in srs.items()) +
            (f"  => 合同[{over}]" if over else "  => 无合同"))
    print("  " + ("✓ 合同市场形成分工" if any_contract else "✗ 无细胞过合同线"))
    print()
    print("[效果]")
    final = m.get("final", {})
    best = m.get("best_ppl", {})
    for k in ("en_lit", "zh_classic", "code_py", "en_wiki"):
        if k in final:
            f = final[k]["ppl"]
            b = best.get(k, f)
            print(f"  {k:12s}: final={f:11.1f}  best={b:11.1f}  "
                  f"forget={f-b:11.1f}")
    print()
    n_cells = m.get("cells_final", len(m.get("territory", [])))
    n_params = m.get("n_params_final", m.get("n_params", 0)) / 1e6
    st = m.get("step_time_mean", 0)
    print(f"[规模/算力] cells={n_cells}  params={n_params:.1f}M  "
          f"step_time={st:.3f}s  wall={m.get('wall_sec',0)/60:.1f}min")
    lit_ok = evk.get("seal", 0) > 0
    div_ok = evk.get("birth", 0) + evk.get("death", 0) > 0
    rep_ok = evk.get("reprise", 0) > 0
    print()
    print("结论: 封存=%s 分化=%s 复现=%s 合同=%s" % (
        "✓" if lit_ok else "✗", "✓" if div_ok else "✗",
        "✓" if rep_ok else "✗", "✓" if any_contract else "✗"))
    print("  => " + ("功能成立" if (lit_ok and any_contract)
                       else "功能未完全成立（机制未充分点火）"))

if __name__ == "__main__":
    verdict(json.load(open(sys.argv[1])))
