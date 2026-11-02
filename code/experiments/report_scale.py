"""report_scale.py —— 规模实验报告生成器（自包含 HTML）。

汇总 results_scale/ 下各规模 run 的 metrics.json：
  * 规模阶梯对比表（参数量 / token 吞吐 / 墙钟 / 终局 ppl / 遗忘）
  * 每域 ppl 曲线（标注 6 段边界与机制事件）
  * 领地热图（细胞 × 概念段的 seg_rel）
  * 机制事件时间线 + 生成样本
  * 与玩具协议（results_lm）的机制同构对照

用法：python3 experiments/report_scale.py
"""
import os, sys, json, glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（容器内常见路径；找不到则退英文标签）
for cand in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/arphic/uming.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(_ROOT, "obm_mvp/results_scale")
UNIQ = ["en_lit", "zh_classic", "code_py", "en_wiki"]
DOM_ZH = {"en_lit": "英文文学", "zh_classic": "中文古典", "code_py": "Python 代码",
          "en_wiki": "英文百科"}
COLORS = {"en_lit": "#0ea5e9", "zh_classic": "#ef4444",
          "code_py": "#10b981", "en_wiki": "#f59e0b"}


def load_runs():
    """汇总 results_scale/ 下各规模 run，排除冒烟/失败/过短 run。

    过滤规则：
      * 目录名含 "smoke" 的纯冒烟 run（如 pilot_smoke，墙钟 <1min，ppl 10^12 噪声）；
      * 墙钟 wall_sec < 300s（5min）视为不完整/失败 run；
      * 无 metrics.json 的空目录（如 pilot_smoke2/3、S1_v1）。
    """
    runs = {}
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        base = os.path.basename(d)
        if "smoke" in base or not os.path.isdir(d):
            continue
        mj = os.path.join(d, "metrics.json")
        if not os.path.exists(mj):
            continue
        try:
            with open(mj) as f:
                m = json.load(f)
        except Exception:
            continue
        if m.get("wall_sec", 0) < 300:
            continue
        runs[base] = m
    return runs


def fig_ppl_curves(m, path):
    """每域 ppl 曲线（对数轴），标注段边界与事件。"""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    spb = m["steps_per_block"]
    for k in UNIQ:
        pts = m["curves"][k]
        xs = [p["step"] for p in pts]
        ys = [max(p["ppl"], 1.0) for p in pts]
        ax.plot(xs, ys, color=COLORS[k], lw=1.6, label=DOM_ZH[k])
    for b in range(1, 6):
        ax.axvline(b * spb, color="#94a3b8", ls=":", lw=1)
    for e in m.get("events_all", []):
        if e[1] == "seal":
            ax.axvline(e[0], color="#334155", ls="--", lw=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("训练步")
    ax.set_ylabel("集成困惑度（对数）")
    ax.set_title(f"OBM-LM {m['scale']}：四域困惑度曲线（虚线=概念段封存）")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_territory(m, path):
    """细胞 × 概念段 seg_rel 热图（领地识别质量）。"""
    cells = m.get("territory", [])
    if not cells:
        return False
    segs = sorted({int(s) for c in cells for s in c["seg_rel"]})
    if not segs:
        return False
    mat = np.full((len(cells), len(segs)), np.nan)
    for i, c in enumerate(cells):
        for s, v in c["seg_rel"].items():
            j = segs.index(int(s))
            mat[i, j] = v
    fig, ax = plt.subplots(figsize=(1.1 * len(segs) + 2.2, 0.62 * len(cells) + 2.2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(segs)), [f"段{s}" for s in segs])
    ax.set_yticks(range(len(cells)), [f"细胞{c['cell']}" for c in cells])
    for i in range(len(cells)):
        for j in range(len(segs)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#1e293b")
    ax.set_title("终局领地账本（细胞 × 概念段可靠性）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def fig_events(m, path):
    """机制事件时间线。"""
    evs = m.get("events_all", [])
    if not evs:
        return False
    kinds = ["seal", "birth", "death", "forget"]
    present = [k for k in kinds if any(e[1] == k for e in evs)]
    other = sorted({e[1] for e in evs} - set(present))
    rows = present + [f"other:{o}" for o in other]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.6))
    for yi, k in enumerate(rows):
        xs = [e[0] for e in evs
              if e[1] == k or f"other:{e[1]}" == k]
        ax.scatter(xs, [yi] * len(xs), s=42, zorder=3,
                   color={"seal": "#334155", "birth": "#16a34a",
                          "death": "#dc2626", "forget": "#7c3aed"}.get(
                              k, "#64748b"))
    ax.set_yticks(range(len(rows)), rows)
    spb = m["steps_per_block"]
    for b in range(6):
        ax.axvline(b * spb, color="#cbd5e1", ls=":", lw=1)
    ax.set_xlabel("训练步")
    ax.set_title("机制事件时间线（封存 / 分裂 / 凋亡 / 挤出）")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def html_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main():
    runs = load_runs()
    if not runs:
        print("no runs found in", OUT)
        return
    figs = {}
    for name, m in runs.items():
        d = os.path.join(OUT, name)
        figs[name] = {}
        p1 = os.path.join(d, "fig_ppl.png")
        fig_ppl_curves(m, p1)
        figs[name]["ppl"] = "fig_ppl.png"
        p2 = os.path.join(d, "fig_territory.png")
        if fig_territory(m, p2):
            figs[name]["territory"] = "fig_territory.png"
        p3 = os.path.join(d, "fig_events.png")
        if fig_events(m, p3):
            figs[name]["events"] = "fig_events.png"

    # ---------- 汇总表 ----------
    rows = []
    for name, m in runs.items():
        fin = m["final"]
        rows.append(dict(
            name=name, scale=m["scale"],
            params=m["n_params_final"] / 1e6,
            d=m["d_model"], L=m["n_layers"], V=m["vocab"],
            tokens=m["tokens"] / 1e6, tok_s=m["tok_per_sec"],
            wall=m["wall_sec"] / 60, cells=m["cells_final"],
            ppls={k: fin[k]["ppl"] for k in UNIQ},
            mean_ppl=float(np.mean([fin[k]["ppl"] for k in UNIQ])),
            forget={k: m["forgetting"].get(k) for k in
                    ("en_lit", "zh_classic")},
        ))
    rows.sort(key=lambda r: r["params"])

    # ---------- HTML ----------
    H = []
    H.append("""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OBM-LM 规模实验报告</title>
<style>
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
max-width:1080px;margin:0 auto;padding:24px;color:#1e293b;line-height:1.65}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{border:1px solid #cbd5e1;padding:6px 8px;text-align:right}
th{background:#f1f5f9}td.l,th.l{text-align:left}
h1{font-size:24px;border-bottom:2px solid #334155;padding-bottom:8px}
h2{font-size:19px;margin-top:34px;color:#0f172a}
h3{font-size:15px;color:#334155}
img{max-width:100%;border:1px solid #e2e8f0;border-radius:6px;margin:8px 0}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
.note{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;
margin:14px 0;font-size:13px}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;
overflow-x:auto;font-size:12px}
</style></head><body>""")
    H.append("<h1>OBM-LM 规模实验报告：从玩具协议到真实语料 LLM 规模</h1>")
    H.append(f"""<p>本报告汇总 <code>results_scale/</code> 下各规模训练 run。
架构与玩具协议完全同构（OBM-LM 多细胞复合体：概念签名 / 条件合同市场 /
双信号边界检测 / 单份授予 + 复现授予 / 梦境代谢），仅把引擎从 numpy
mini-autograd 换成 PyTorch（bf16 autocast + 8 线程甜点），语料换成
四域真实文本（英文文学 / 中文古典 / Python 代码 / 英文百科，BPE-16K 分词）。
六段协议：4 域顺序 + 后两段复现（测抗遗忘与复现授予）。</p>""")

    H.append("<h2>1. 规模阶梯总览</h2>")
    H.append("<table><tr><th class='l'>run</th><th>D</th><th>层</th>"
             "<th>总参数</th><th>细胞数</th><th>训练 tokens</th>"
             "<th>tok/s</th><th>墙钟</th><th>四域均值 ppl</th></tr>")
    for r in rows:
        H.append(f"<tr><td class='l'>{r['name']}</td><td>{r['d']}</td>"
                 f"<td>{r['L']}</td><td>{r['params']:.1f}M</td>"
                 f"<td>{r['cells']}</td><td>{r['tokens']:.1f}M</td>"
                 f"<td>{r['tok_s']:.0f}</td><td>{r['wall']:.0f}min</td>"
                 f"<td>{r['mean_ppl']:.1f}</td></tr>")
    H.append("</table>")

    for name, m in runs.items():
        r = next(x for x in rows if x["name"] == name)
        H.append(f"<h2>2. {name}（{m['scale']}，D={m['d_model']}，"
                 f"{m['n_layers']} 层，{r['params']:.1f}M 参数，"
                 f"{m['tokens']/1e6:.1f}M tokens，{r['wall']:.0f} 分钟）</h2>")
        H.append(f"<img src='{name}/{figs[name]['ppl']}'>")
        fin = m["final"]
        H.append("<table><tr><th class='l'>域</th><th>终局 ppl</th>"
                 "<th>全程最佳 ppl</th><th>遗忘量 (final−best)</th></tr>")
        for k in UNIQ:
            fg = m["forgetting"].get(k)
            fg_s = f"{fg:+.1f}" if fg is not None else "—"
            H.append(f"<tr><td class='l'>{DOM_ZH[k]}（{k}）</td>"
                     f"<td>{fin[k]['ppl']:.1f}</td>"
                     f"<td>{m['best_ppl'][k]:.1f}</td><td>{fg_s}</td></tr>")
        H.append("</table>")
        H.append(f"<p>复现段恢复速度（复现域首个评估点 ppl vs 最佳）："
                 + "；".join(f"{DOM_ZH[k]} {v:.1f}"
                             for k, v in (m.get('reprise_first_eval') or {}).items()
                             if v is not None) + "。</p>")
        if figs[name].get("events"):
            H.append(f"<img src='{name}/{figs[name]['events']}'>")
        if figs[name].get("territory"):
            H.append(f"<img src='{name}/{figs[name]['territory']}'>")
        # 路由诊断（若有 diag_routing.json）：合同市场是否把输入送到对的细胞
        drp = os.path.join(OUT, name, "diag_routing.json")
        if os.path.exists(drp):
            try:
                dr = json.load(open(drp))
                H.append("<h3>路由诊断：合同市场是否按概念签名分工</h3>")
                H.append("<table><tr><th class='l'>域</th>"
                         "<th>线上路由 ppl</th><th>全细胞等权</th>"
                         "<th>单细胞最优</th><th>最优细胞</th></tr>")
                for r in dr["rows"]:
                    dom = DOM_ZH.get(r["domain"], r["domain"])
                    H.append(
                        f"<tr><td class='l'>{dom}（{r['domain']}）</td>"
                        f"<td>{r['routed']:.1f}</td>"
                        f"<td>{r['allcells']:.1f}</td>"
                        f"<td>{r['best1']:.1f}</td>"
                        f"<td>c{r['best_cell']}</td></tr>")
                H.append("</table>")
                H.append("<p>解读：若『线上路由』≈『单细胞最优』≪『全细胞等权』，"
                         "说明合同市场真的在执行分工（把每域输入送到对的细胞），"
                         "而非放弃路由的混编。routed 优于 best1 表示该域的集成"
                         "加权进一步压低了困惑度。</p>")
            except Exception:
                pass
        sp = os.path.join(OUT, name, "samples.txt")
        if os.path.exists(sp):
            txt = open(sp).read()
            H.append("<h3>生成样本（集成路由贪心解码）</h3>")
            H.append(f"<pre>{html_escape(txt[:2200])}</pre>")
        evs = m.get("events_all", [])
        if evs:
            log_txt = "\n".join(f"[{s:5d}] {k:6s} {msg}"
                                for s, k, msg in evs[:80])
            H.append("<h3>机制事件日志</h3><pre>" + html_escape(log_txt)
                     + "</pre>")
        H.append("")

    H.append("""<h2>3. 工程注记（诚实记录）</h2>
<div class="note"><b>性能</b>：V=16K 下初版 run_step 8.5s/步。三处修复：
(a) 路由统计前向融合为单趟（autocast + 分块 softmax/max/argmax 合一，
logsumexp 全量物化是单项 1.75s/步的病灶）；(b) decide 的集成统计改在
torch 内计算（旧实现把每个细胞的 (24,256,16384) softmax 搬进 numpy，
1.6s/步）；(c) 熵缓存刷新回归 dist_refresh=10 间隔制（每步强制刷新是
移植期遗留，玩具协议下两种刷新频率决策逐位等价——已对照验证）。
合计 8.5s/步 -> 1.25s/步（6.8×）。</div>
<div class="note"><b>checkpoint</b>：初版三类静默污染——(a) numpy 数组
与活参数共享存储（.numpy()/from_numpy 浅引用），原模型训练经 Adam 就地
更新改写恢复体；(b) cur_x/events/history/segments 浅引用，保存后继续
训练污染快照；(c) 路由统计缓存（上一样本统计）未入库导致恢复体选路
分歧。修复后双引擎（numpy/torch）均实现恢复后逐步位一致（含 pickle 落盘
往返），Adam 一阶/二阶矩与 rng 状态完整入库。</div>
<div class="note"><b>bf16 autocast</b>：CPU AVX512-BF16 下矩阵乘激活值
降精度、参数与 Adam 保持 fp32，实测 1.48× 加速（D=384）。</div>
<div class="note"><b>路由坍缩 bug 与修复（核心发现）</b>：pilot 终局诊断暴露
合同市场坍缩到单一通才细胞 c2——6/6 评估批全选、权重 1.00。code_py 线上
ppl 高达 <b>97861</b>，而 code 专才 c1 的单细胞 ppl 仅 1332（73× 差距）；
en_wiki 48850 vs 专才 3401。三大病灶：(a) 未训练段的
<code>cond_reliability</code> 先验 0.5 让「从未受训」≈「专才」；(b) 合同池空
时回退到<b>陈旧 probe 缓存</b>（上次训练批）做彩票授予，eval/生成路径
张冠李戴；(c) 难域绝对精度达不到记账线 0.12，专才 seg_rel 磨到 0 过不了
合同线。修复：未知段先验改 0（未证明=不可靠）、探针彩票门禁到训练路径
（eval/generate 改按 cond_rel 排序取 top-k）、引入段内峰值参考。
<b>零重训练、纯换选路逻辑</b>：code_py 97861→1141（86×↓）、en_wiki
48850→3842（13×↓），routed 与单细胞最优对齐；en_lit/zh 路由到其天然归属
细胞 c2（owner 不变，集成轻微稀释）。详见各 run 的「路由诊断」表。
<code>experiments/diag_routing.py</code> 复现该诊断。</div>
<div class="note"><b>玩具回归守卫</b>：<code>experiments/toy_regress.py</code>
（numpy 路，6 段 × 300 步）在每次改动后必跑——终局精度、封存/分裂/凋亡/
复现授予事件数、复现段抗遗忘三条底线。<code>experiments/torch_smoke.py</code>
覆盖 torch 路全机制（梦境/评估/checkpoint/生成）。</div>""")

    H.append("</body></html>")
    out = os.path.join(OUT, "report_scale.html")
    with open(out, "w") as f:
        f.write("\n".join(H))
    print("wrote", out)


if __name__ == "__main__":
    main()
