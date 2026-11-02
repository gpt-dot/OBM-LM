"""report_lab.py —— 跳跃型迭代对标报告：速度 × 功能 × 内存。

读 results_lab/*.json（obm_lab 产出），以 pilot_v1 基线（或 half 基线）
为 1.0×，按 step_time 提速排序，功能判据（seal/birth/death）单独成列。
绝不把"功能 ✗"的配置排成成功——pass=False 的行标红。
"""
import json, glob, os, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

LAB = os.path.join(_ROOT, "obm_mvp/results_lab")
# 基线：pilot_v1（run_scale 产出）的 step_time 用于算总提速
PILOT_STEP = None


def load_all():
    rows = []
    for f in sorted(glob.glob(os.path.join(LAB, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "step_time_mean" not in d:
            continue
        rows.append(d)
    return rows


def fmt(v, nd=0):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-step", type=float, default=1.94,
                    help="pilot_v1 基线 step_time(s)，用于算总提速")
    args = ap.parse_args()
    rows = load_all()
    if not rows:
        print("no lab results yet"); return
    rows.sort(key=lambda d: d["step_time_mean"])
    base = args.baseline_step

    # ---- 文本表 ----
    hdr = (f"{'tag':<14}{'D':>4}{'L':>3}{'B':>4}{'ctx':>5}"
           f"{'params(M)':>10}{'step(ms)':>9}{'speedup':>9}"
           f"{'tok/s':>8}{'RSS(MB)':>9}{'cells':>6}{'seg':>5}  "
           f"{'seal':>5}{'birth':>6}{'death':>6}{'PASS':>6}")
    lines = [hdr, "-" * len(hdr)]
    for d in rows:
        kc = d.get("event_counts", {})
        sp = base / d["step_time_mean"]
        lines.append(
            f"{d['tag']:<14}{d['d_model']:>4}{d['n_layers']:>3}{d['batch']:>4}"
            f"{d.get('ctx', 256):>5}{d['n_params']/1e6:>10.1f}"
            f"{d['step_time_mean']*1000:>9.0f}{sp:>8.2f}x"
            f"{d['tok_per_sec']:>8.0f}{d['peak_rss_mb']:>9.0f}"
            f"{d['cells_final']:>6}{d.get('segments', 0):>5}  "
            f"{kc.get('seal', 0):>5}{kc.get('birth', 0):>6}{kc.get('death', 0):>6}"
            f"{('✓' if d.get('function_pass') else '✗'):>6}")
    table = "\n".join(lines)
    print(table)

    # ---- 找最快且功能 pass 的配置 ----
    ok = [d for d in rows if d.get("function_pass")]
    best = ok[0] if ok else None
    if best:
        print(f"\n★ 最快且功能合格：{best['tag']}  "
              f"step={best['step_time_mean']*1000:.0f}ms  "
              f"speedup={base/best['step_time_mean']:.2f}×  "
              f"RSS={best['peak_rss_mb']:.0f}MB  events={best['event_counts']}")

    # ---- HTML ----
    def row_html(d):
        kc = d.get("event_counts", {})
        sp = base / d["step_time_mean"]
        cls = "pass" if d.get("function_pass") else "fail"
        return (f"<tr class='{cls}'><td>{d['tag']}</td>"
                f"<td>{d['d_model']}</td><td>{d['n_layers']}</td>"
                f"<td>{d['batch']}</td><td>{d.get('ctx',256)}</td>"
                f"<td>{d['n_params']/1e6:.1f}M</td>"
                f"<td>{d['step_time_mean']*1000:.0f}</td>"
                f"<td><b>{sp:.2f}×</b></td>"
                f"<td>{d['tok_per_sec']:.0f}</td><td>{d['peak_rss_mb']:.0f}</td>"
                f"<td>{d['cells_final']}</td><td>{d.get('segments',0)}</td>"
                f"<td>{kc.get('seal',0)}</td><td>{kc.get('birth',0)}</td>"
                f"<td>{kc.get('death',0)}</td>"
                f"<td>{'✓' if d.get('function_pass') else '✗ 功能未过'}</td></tr>")
    body = "\n".join(row_html(d) for d in rows)
    best_html = (f"<p class='best'>★ 最快且功能合格：<b>{best['tag']}</b> — "
                 f"{base/best['step_time_mean']:.2f}× 提速，"
                 f"step {best['step_time_mean']*1000:.0f}ms，"
                 f"RSS {best['peak_rss_mb']:.0f}MB，事件 {best['event_counts']}</p>"
                 ) if best else "<p class='best'>暂无功能合格的配置</p>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<style>
body{{font-family:system-ui,'Segoe UI',sans-serif;margin:24px;background:#0f1117;color:#e6e6e6}}
h1{{font-size:20px}} table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}}
th,td{{border:1px solid #2a2f3a;padding:6px 8px;text-align:right}}
th{{background:#1a1e28;text-align:center}} td:first-child,th:first-child{{text-align:left}}
tr.pass td{{background:#12271a}} tr.fail td{{background:#2a1616;opacity:.75}}
.best{{background:#14301f;border:1px solid #2e6b3f;padding:10px 14px;border-radius:8px;font-size:15px}}
.note{{color:#8a92a6;font-size:12px;margin-top:10px}}
</style></head><body>
<h1>OBM-LM 跳跃型迭代对标（速度 × 功能 × 内存）</h1>
{best_html}
<table><tr><th>配置</th><th>D</th><th>L</th><th>B</th><th>ctx</th><th>参数</th>
<th>step(ms)</th><th>提速</th><th>tok/s</th><th>RSS(MB)</th><th>细胞</th><th>段</th>
<th>封存</th><th>分裂</th><th>凋亡</th><th>功能</th></tr>
{body}
</table>
<p class='note'>基线 pilot_v1 step={base*1000:.0f}ms。提速=pilot/本配置。
绿行=功能合格（封存点火）；红行=功能未过（机制未点火，不计成功）。
生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}。</p>
</body></html>"""
    out = os.path.join(LAB, "report_lab.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"\n[report] {out}")


if __name__ == "__main__":
    main()
