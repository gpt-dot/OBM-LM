"""run_all.py —— OBM MVP 四组对照实验（多种子重复）。

对照组：
  A. OBM 完整版      分歧路由 + 梦境代谢 + 合同市场 + 三因子学习
  B. OBM 无梦境      dream_enabled=False（消融 M4：检验抗遗忘）
  C. OBM 相似路由    router="similarity"（消融 M2：检验分歧路由的价值）
  D. SingleNet       单 MLP(≈同等参数量) + 全局反传 + 在线 SGD（传统范式对照物）

每个配置跑 SEEDS 个随机种子，报告均值 ± 标准差。
运行：python3 -m experiments.run_all   （在 obm_mvp/ 目录下）
"""
import base64
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.baselines import SingleNet
from obm.stream import build_stream

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PLAN = ["linear", "xor", "wave", "blobs3", "linear", "xor", "moons"]
BLOCK = 600
EVAL_EVERY = 50
SEEDS = [1, 2, 3, 4, 5]
CONFIGS = ["A · OBM 完整版", "B · OBM 无梦境", "C · OBM 相似路由", "D · SingleNet 基线"]


def seen_task_indices(step):
    block = min(step // BLOCK, len(PLAN) - 1)
    uniq = []
    for k in PLAN[: block + 1]:
        if k not in uniq:
            uniq.append(k)
    order = list(dict.fromkeys(PLAN))
    return [order.index(k) for k in uniq]


def make_model(name, seed):
    if name.startswith("D"):
        return SingleNet(n_classes=3, seed=seed)
    cfg = OBMConfig(seed=seed)
    if "无梦境" in name:
        cfg.dream_enabled = False
    if "相似路由" in name:
        cfg.router = "similarity"
    return OuroborosMesh(cfg)


def run_config(name, model, train_blocks, test_sets, names):
    total_steps = sum(len(y) for _, y in train_blocks)
    online_correct, online_k, online_params = [], [], []
    eval_points, eval_means, eval_matrix, eval_seen = [], [], [], []
    reflex_flags = []
    per_task_best = {}
    step = 0
    for X, y in train_blocks:
        for x, yy in zip(X, y):
            rec = model.run_step(x, yy)
            online_correct.append(rec["correct"])
            online_k.append(rec["k"])
            online_params.append(rec["n_params"])
            reflex_flags.append(rec.get("reflex", 0))
            if step % EVAL_EVERY == 0 or step == total_steps - 1:
                idxs = seen_task_indices(step)
                accs = model.evaluate([test_sets[i] for i in idxs])
                eval_points.append(step)
                eval_seen.append(idxs)
                eval_means.append(float(np.mean(accs)))
                eval_matrix.append(accs)
                for i, a in zip(idxs, accs):
                    per_task_best[i] = max(per_task_best.get(i, 0.0), a)
            step += 1
    final_accs = model.evaluate(test_sets)
    final_mean = float(np.mean(final_accs))
    forgetting = float(np.mean([per_task_best.get(i, a) - a
                                for i, a in enumerate(final_accs)]))
    starts = np.cumsum([0] + [len(y) for _, y in train_blocks])
    revisit_recovery = float(np.mean([
        np.mean(online_correct[starts[b]:starts[b] + 100]) for b in (4, 5)]))
    ev = getattr(model, "events", [])
    n_ev = lambda t: sum(1 for e in ev if e[1] == t)
    summary = dict(
        name=name,
        final_accs=[float(a) for a in final_accs],
        final_mean_acc=final_mean,
        forgetting=forgetting,
        online_acc_last500=float(np.mean(online_correct[-500:])),
        revisit_recovery=revisit_recovery,
        total_energy=model.energy,
        energy_per_decision=model.energy / total_steps,
        final_params=int(online_params[-1]),
        reflex_ratio_late=float(np.mean(reflex_flags[-400:])),
        n_seals=n_ev("seal"), n_births=n_ev("birth"), n_deaths=n_ev("death"),
    )
    curves = dict(online_correct=online_correct, online_k=online_k,
                  online_params=online_params, reflex=reflex_flags,
                  eval_points=eval_points, eval_means=eval_means,
                  eval_matrix=eval_matrix, eval_seen=eval_seen)
    return summary, curves, ev


def moving_avg(xs, w=100):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < w:
        return xs
    c = np.cumsum(np.insert(xs, 0, 0))
    ma = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), ma])


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    os.makedirs(OUT, exist_ok=True)
    train_blocks, test_sets, names = build_stream()
    total_steps = sum(len(y) for _, y in train_blocks)
    print(f"[stream] {len(train_blocks)} 个概念段，共 {total_steps} 步；概念: {names}")
    print(f"[seeds] {SEEDS}")

    per_seed = {n: {} for n in CONFIGS}
    for name in CONFIGS:
        for seed in SEEDS:
            model = make_model(name, seed)
            s, c, e = run_config(name, model, train_blocks, test_sets, names)
            per_seed[name][seed] = (s, c, e)
        ms = [per_seed[name][sd][0]["final_mean_acc"] for sd in SEEDS]
        fs = [per_seed[name][sd][0]["forgetting"] for sd in SEEDS]
        print(f"[done] {name}: 最终均值 {np.mean(ms):.3f}±{np.std(ms):.3f} | "
              f"遗忘 {np.mean(fs):.3f}±{np.std(fs):.3f}")

    # ---------- 聚合 ----------
    agg = {}
    for name in CONFIGS:
        ss = [per_seed[name][sd][0] for sd in SEEDS]
        a = {}
        for key in ["final_mean_acc", "forgetting", "online_acc_last500",
                    "revisit_recovery", "energy_per_decision",
                    "reflex_ratio_late", "final_params", "n_seals",
                    "n_births", "n_deaths"]:
            a[key] = mean_std([s[key] for s in ss])
        a["final_accs"] = [mean_std([s["final_accs"][i] for s in ss])
                           for i in range(len(names))]
        agg[name] = a
        agg[name]["events_seed1"] = per_seed[name][SEEDS[0]][2]

    def curve(name, key, w=None):
        arrs = []
        for sd in SEEDS:
            xs = per_seed[name][sd][1][key]
            arrs.append(moving_avg(xs, w) if w else np.asarray(xs, dtype=float))
        arrs = np.array(arrs)
        return np.nanmean(arrs, axis=0), np.nanstd(arrs, axis=0)

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            names=names, total_steps=total_steps, seeds=SEEDS,
            agg={n: {k: list(v) for k, v in agg[n].items() if k != "events_seed1"}
                 for n in CONFIGS},
            events={n: per_seed[n][SEEDS[0]][2] for n in CONFIGS},
        ), f, ensure_ascii=False, indent=2)

    boundaries = [i * BLOCK for i in range(len(PLAN) + 1)]
    colors = {"A · OBM 完整版": "#1d4ed8", "B · OBM 无梦境": "#dc2626",
              "C · OBM 相似路由": "#d97706", "D · SingleNet 基线": "#64748b"}
    full = "A · OBM 完整版"

    def band(ax, m, s, color, label, lw=1.8):
        ax.plot(m, color=color, lw=lw, label=label)
        ax.fill_between(range(len(m)), m - s, m + s, color=color, alpha=.13, lw=0)

    # ---- 图 1：在线精度 ----
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    for name in CONFIGS:
        m, s = curve(name, "online_correct", w=100)
        band(ax, m, s, colors[name], name)
    for b in boundaries[1:-1]:
        ax.axvline(b, color="#cbd5e1", ls="--", lw=1)
    for i, k in enumerate(PLAN):
        ax.text(i * BLOCK + BLOCK / 2, 1.02, k, ha="center", fontsize=8.5, color="#64748b")
    ax.set_xlabel("时间步（概念段顺序到达，竖线为概念切换）")
    ax.set_ylabel("在线精度（100 步滑动平均）")
    ax.set_title("图 1 · 在线精度（5 种子均值 ± 标准差）")
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_online.png"))
    plt.close(fig)

    # ---- 图 2：持续学习保持率 ----
    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=130)
    task_colors = {"linear": "#0ea5e9", "xor": "#8b5cf6", "wave": "#f59e0b",
                   "blobs3": "#10b981", "moons": "#ef4444"}
    for ti, tname in enumerate(names):
        def task_series(cur):
            pts = [(p, row[idxs.index(ti)]) for p, row, idxs in
                   zip(cur["eval_points"], cur["eval_matrix"], cur["eval_seen"])
                   if ti in idxs]
            return [p for p, _ in pts], [v for _, v in pts]
        xs, _ = task_series(per_seed[full][SEEDS[0]][1])
        ys = np.zeros((len(SEEDS), len(xs)))
        for si, sd in enumerate(SEEDS):
            _, vals = task_series(per_seed[full][sd][1])
            ys[si] = vals
        ax.plot(xs, ys.mean(0), color=task_colors[tname], lw=1.3, alpha=.85,
                label=f"任务 {tname}")
        ax.fill_between(xs, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0),
                        color=task_colors[tname], alpha=.10, lw=0)
    for name in CONFIGS:
        em = np.array([per_seed[name][sd][1]["eval_means"] for sd in SEEDS])
        xs = per_seed[name][SEEDS[0]][1]["eval_points"]
        lw = 3 if name == full else 1.8
        ls = "-" if name == full else "--"
        ax.plot(xs, em.mean(0), color=colors[name], lw=lw, ls=ls,
                label=f"{name}（均值）")
        ax.fill_between(xs, em.mean(0) - em.std(0), em.mean(0) + em.std(0),
                        color=colors[name], alpha=.10, lw=0)
    for b in boundaries[1:-1]:
        ax.axvline(b, color="#cbd5e1", ls="--", lw=1)
    ax.set_xlabel("时间步")
    ax.set_ylabel("测试精度")
    ax.set_title("图 2 · 持续学习保持率：细线=完整版 OBM 的各任务精度，组均值见粗/虚线")
    ax.set_ylim(0.25, 1.05)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_retention.png"))
    plt.close(fig)

    # ---- 图 3：新陈代谢 ----
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    ax2 = ax.twinx()
    m, s = curve(full, "online_params")
    ax.plot(m, color="#0f766e", lw=2, label="总参数量（均值）")
    ax.fill_between(range(len(m)), m - s, m + s, color="#0f766e", alpha=.13, lw=0)
    cell_counts, counts = [], 3
    ev_iter = iter(agg[full]["events_seed1"])
    next_ev = next(ev_iter, None)
    for st in range(total_steps):
        while next_ev is not None and next_ev[0] <= st:
            if next_ev[1] == "birth":
                counts += 1
            elif next_ev[1] in ("death", "merge"):
                counts -= 1
            next_ev = next(ev_iter, None)
        cell_counts.append(counts)
    ax2.plot(cell_counts, color="#6d28d9", lw=1.6, label="细胞数量（种子 1）")
    for st, typ, _ in agg[full]["events_seed1"]:
        if typ == "birth":
            ax.axvline(st, color="#16a34a", ls=":", lw=1.2)
        elif typ in ("death", "merge"):
            ax.axvline(st, color="#dc2626", ls=":", lw=1.2)
        elif typ == "seal":
            ax.axvline(st, color="#94a3b8", ls=":", lw=0.7)
    ax.set_xlabel("时间步")
    ax.set_ylabel("总参数量", color="#0f766e")
    ax2.set_ylabel("细胞数量", color="#6d28d9")
    ax.set_title("图 3 · 梦境代谢：绿线=分裂诞生，红线=凋亡回收，参数总量始终有界")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_metabolism.png"))
    plt.close(fig)

    # ---- 图 4：能耗与快慢双系统 ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=130)
    w = 50
    mk, sk = curve(full, "online_k", w=w)
    axes[0].plot(mk, color="#1d4ed8", lw=1.6, label="均值")
    axes[0].fill_between(range(len(mk)), mk - sk, mk + sk, color="#1d4ed8",
                         alpha=.13, lw=0)
    for b in boundaries[1:-1]:
        axes[0].axvline(b, color="#cbd5e1", ls="--", lw=1)
    axes[0].set_title("每步激活的细胞数（50 步窗口）")
    axes[0].set_xlabel("时间步")
    axes[0].set_ylabel("平均激活细胞数")
    axes[0].grid(alpha=.25)
    mr, sr = curve(full, "reflex", w=w)
    axes[1].plot(mr, color="#0f766e", lw=1.6)
    axes[1].fill_between(range(len(mr)), mr - sr, mr + sr, color="#0f766e",
                         alpha=.13, lw=0)
    for b in boundaries[1:-1]:
        axes[1].axvline(b, color="#cbd5e1", ls="--", lw=1)
    axes[1].set_title("反射弧触发比例（越高越省能耗）")
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("反射比例")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=.25)
    fig.suptitle("图 4 · 快慢双系统：简单任务走反射弧，新概念出现时自动深思", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_energy.png"))
    plt.close(fig)

    # ---- 图 5：总结 ----
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), dpi=130)
    metrics = [("final_mean_acc", "最终平均精度"),
               ("forgetting", "遗忘量（越低越好）"),
               ("energy_per_decision", "每决策参数激活量"),
               ("reflex_ratio_late", "后期反射弧比例")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [agg[n][key][0] for n in CONFIGS]
        errs = [agg[n][key][1] for n in CONFIGS]
        bars = ax.bar(range(len(CONFIGS)), vals, yerr=errs, capsize=3,
                      color=[colors[n] for n in CONFIGS], alpha=.85)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([n.split(" ")[0] for n in CONFIGS], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(np.array(vals) + np.array(errs)) * 1.3 + 1e-9)
    fig.suptitle(f"图 5 · 四组实验总结（{len(SEEDS)} 种子均值 ± 标准差）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_summary.png"))
    plt.close(fig)

    # ---- 报告 ----
    imgs = {}
    for f in ["fig1_online.png", "fig2_retention.png", "fig3_metabolism.png",
              "fig4_energy.png", "fig5_summary.png"]:
        with open(os.path.join(OUT, f), "rb") as fh:
            imgs[f] = base64.b64encode(fh.read()).decode()

    rows = ""
    for n in CONFIGS:
        a = agg[n]
        rows += (f"<tr><td><b>{n}</b></td>"
                 f"<td>{a['final_mean_acc'][0]:.3f}±{a['final_mean_acc'][1]:.3f}</td>"
                 f"<td>{a['forgetting'][0]:.3f}±{a['forgetting'][1]:.3f}</td>"
                 f"<td>{a['revisit_recovery'][0]:.3f}±{a['revisit_recovery'][1]:.3f}</td>"
                 f"<td>{a['final_params'][0]:.0f}</td>"
                 f"<td>{a['energy_per_decision'][0]:.0f}±{a['energy_per_decision'][1]:.0f}</td>"
                 f"<td>{a['reflex_ratio_late'][0]:.2f}</td>"
                 f"<td>{a['n_seals'][0]:.0f}/{a['n_births'][0]:.0f}/{a['n_deaths'][0]:.0f}</td></tr>")

    per_task = "<table><tr><th>配置</th>" + "".join(f"<th>{t}</th>" for t in names) + "</tr>"
    for n in CONFIGS:
        cells = "".join(f"<td>{m:.2f}<sub>±{s:.2f}</sub></td>"
                        for m, s in agg[n]["final_accs"])
        per_task += f"<tr><td><b>{n}</b></td>{cells}</tr>"
    per_task += "</table>"

    ev_rows = ""
    for st, typ, detail in agg[full]["events_seed1"][:50]:
        ev_rows += f"<tr><td>{st}</td><td>{typ}</td><td>{detail}</td></tr>"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>OBM MVP 实验报告</title>
<style>
body{{font-family:"Noto Sans CJK SC","PingFang SC",sans-serif;background:#f1f5f9;color:#1e293b;
max-width:1000px;margin:0 auto;padding:36px 20px;line-height:1.7}}
h1{{font-size:26px}} h2{{font-size:19px;margin-top:34px;border-left:5px solid #1d4ed8;padding-left:10px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0}}
th{{background:#0f172a;color:#fff;padding:8px 10px;text-align:left}}
td{{border:1px solid #e2e8f0;padding:7px 10px}} tr:nth-child(even) td{{background:#f8fafc}}
img{{width:100%;border:1px solid #e2e8f0;border-radius:10px;margin:10px 0;background:#fff}}
.note{{background:#eff6ff;border-left:4px solid #1d4ed8;padding:12px 16px;border-radius:0 8px 8px 0;font-size:14px}}
code{{background:#e2e8f0;padding:1px 6px;border-radius:4px;font-size:13px}}
</style></head><body>
<h1>衔尾蛇 OBM · 最小可行产品（MVP）实验报告</h1>
<p>概念漂移持续学习流：7 个概念段顺序到达（linear → xor → wave → blobs3 → <b>linear 复现</b> → <b>xor 复现</b> → moons），
共 {total_steps} 步，每配置 {len(SEEDS)} 个随机种子。系统不知道段边界（OBM 用性能跌落检测自己发现）。
遗忘量 = 各任务"历史最佳精度 − 最终精度"的均值；能耗按<b>参数激活量</b>计（OBM 稀疏激活 vs 基线全参数激活）。</p>

<h2>1. 总体结果（均值 ± 标准差）</h2>
<table><tr><th>配置</th><th>最终平均精度</th><th>遗忘量↓</th><th>复现恢复速度</th><th>最终参数量</th><th>每决策参数激活量↓</th><th>后期反射比例</th><th>封存/分裂/凋亡</th></tr>{rows}</table>
<div class="note"><b>读法：</b>A 与 B 只差"是否做梦"；A 与 C 只差路由方式；D 是同等参数量级、使用全局反向传播的传统 MLP。
遗忘量的组间差异即各机制的净贡献。</div>

<h2>2. 各任务最终精度</h2>
{per_task}

<h2>3. 在线精度</h2>
<img src="data:image/png;base64,{imgs['fig1_online.png']}">

<h2>4. 持续学习保持率（核心证据）</h2>
<img src="data:image/png;base64,{imgs['fig2_retention.png']}">

<h2>5. 梦境代谢：参数总量有界</h2>
<img src="data:image/png;base64,{imgs['fig3_metabolism.png']}">
<table><tr><th>步</th><th>类型</th><th>详情</th></tr>{ev_rows}</table>

<h2>6. 快慢双系统与能耗</h2>
<img src="data:image/png;base64,{imgs['fig4_energy.png']}">

<h2>7. 总结</h2>
<img src="data:image/png;base64,{imgs['fig5_summary.png']}">

<h2>8. 复现方式</h2>
<p><code>cd obm_mvp &amp;&amp; python3 -m experiments.run_all</code>（纯 numpy + matplotlib，无深度学习框架依赖，约 1 分钟）</p>
</body></html>"""
    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("[done] 图表与报告已生成 ->", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
