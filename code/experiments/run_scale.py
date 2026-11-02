"""run_scale.py —— OBM-LM 规模实验：真实文本域流上的持续学习。

协议（与玩具实验 langs.build_blocks 同构的 6 段流）：
    [en_lit, zh_classic, code_py, en_wiki, en_lit*, zh_classic*]
    后两段复现前两个域——测抗遗忘 + 复现恢复。系统靠双信号自行发现段边界，
    段边界不由脚本告知（block 切换只换数据流）。

指标（真实 LM 用困惑度为主，精度被 16K 词表钝化）：
- 每域留出窗口的集成 ppl / next-token 精度曲线（每 eval_every 步全域评估）
- forgetting = 复现域 (final_ppl - best_ppl) 的均值（ppl 越低越好，
  正数=退化）
- reprise 恢复 = 复现段前 100 步的 ppl 均值 vs 该域专才平台 ppl
- 细胞数/分裂/凋亡/封存/复现授予事件流、吞吐、总参数量
- 终局各域采样（集成路由贪心解码）

阈值标定依据（/tmp/calibrate.py 实测，D=384 签名空间）：
- 跨域 batch 签名距离 19-38 vs 同域 0.5-1.1 -> d_thr=3.0 原值成立，
  award_reprise_d=3.0 覆盖同域复现（max 1.85）且远离跨域
- 段内 EW 距离 0.2-0.7 -> 信号 B 无假阳性
- 精度门限按 pilot 实测重标（fulfill/expert/reprise_probe）

用法：
    python3 experiments/run_scale.py --scale pilot [--steps 300] [--seed 0]
    python3 experiments/run_scale.py --scale S1 --resume <ckpt>
"""
import os, sys, time, json, argparse, pickle, math

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
torch.set_num_threads(8)      # 小矩阵下 8 线程是甜点位（16/32 线程抖动更慢，
#                              见吞吐标定：D=384 8 线程 4.41s vs 16 线程 7.28s）
torch.set_num_threads(int(os.environ.get("OBM_THREADS", "8")))

from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm import corpus
from obm_lm.tokenizer import BPETokenizer

# 根目录可经环境变量 OBM_ROOT 覆盖（默认本地沙箱路径），便于远端部署
_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_OBM_ROOT, "data", "cache")

# ================= 规模阶梯 =================
# 阈值说明（玩具标定值在真实文本下的迁移）：
#  fulfill/expert_acc : next-token 精度平台上，fulfill≈0.35×平台、expert≈0.8×平台
#                       （专才典型精度需高于 expert 线，seg_rel 才爬得上 0.9）
#  award_reprise_probe: 同域专才 ~0.3-0.45 / 跨域 ~0.05-0.15 -> 0.20
#  split_rel_frac=0.85: 按峰值比例判分裂（绝对线 0.95 在真实文本下恒真）
#  acc_prior=0.0       : 峰值从首步跟随实测（0.34 的先验高于早期精度会造成假跌落）
#  specialist_line    : pilot 用玩具值 0.80（保守基线）；S1+ 用 0.60——规模下
#                       seg_rel 平衡点实测 0.58-0.68（pilot s999：最高 0.584），
#                       0.80 永远够不到 -> 复现授予/专才保护全部惰性。0.60 落在
#                       "好细胞平衡点之上、边缘细胞（0.21）之上"的分离带内。
#  内存约束（cgroup 8GB，两训练进程共享会 OOM——S1 曾被杀）：
#   S2/S3 缩到 3 细胞起步（优化器状态 = 4×参数字节：params+grad+adam m+v）。
#   S2: 3×23.9M×16B≈1.2GB + 激活 ~2GB；S3: 3×84M×16B≈4GB + 激活 ~1.2GB。
SCALES = {
    "pilot": dict(d_model=256, n_heads=8, n_layers=4, batch=24, ctx=256,
                  lr=1.2e-3, n_cells_init=4, max_cells=6, specialist_line=0.80,
                  fulfill_acc=0.12, expert_acc=0.30,
                  dream_batches=8, dream_batch=16, buffer_size=600),
    "S1": dict(d_model=384, n_heads=6, n_layers=6, batch=24, ctx=256,
               lr=6e-4, n_cells_init=4, max_cells=8, specialist_line=0.60,
               fulfill_acc=0.12, expert_acc=0.30,
               dream_batches=8, dream_batch=16, buffer_size=600),
    "S2": dict(d_model=512, n_heads=8, n_layers=8, batch=16, ctx=256,
               lr=4e-4, n_cells_init=3, max_cells=5, specialist_line=0.60,
               fulfill_acc=0.12, expert_acc=0.30,
               dream_batches=6, dream_batch=12, buffer_size=600),
    "S3": dict(d_model=768, n_heads=12, n_layers=10, batch=12, ctx=256,
               lr=3e-4, n_cells_init=3, max_cells=5, specialist_line=0.60,
               fulfill_acc=0.12, expert_acc=0.30,
               dream_batches=6, dream_batch=12, buffer_size=600),
}


def build_mesh(scale, seed, vocab):
    s = SCALES[scale]
    cfg = OBMLMConfig(
        seed=seed, backend="torch", vocab=vocab,
        d_model=s["d_model"], n_heads=s["n_heads"], n_layers=s["n_layers"],
        max_len=s["ctx"], batch=s["batch"],
        n_cells_init=s["n_cells_init"], max_cells=s["max_cells"],
        out_full=True, acc_prior=0.0, amp_bf16=True,
        grad_clip=1.0, weight_decay=0.01,
        lr=s["lr"], lr_breach=s["lr"] / 3, dream_lr=s["lr"],
        fulfill_acc=s["fulfill_acc"], expert_acc=s["expert_acc"],
        specialist_line=s["specialist_line"],
        award_reprise_d=3.0, award_reprise_probe=0.20,
        split_rel_frac=0.85, split_cooldown=400,
        dream_batches=s["dream_batches"], dream_batch=s["dream_batch"],
        buffer_size=s["buffer_size"], seg_buffer_size=120,
        gauss_chi=0.0,          # 按 3.9×d_model 自动缩放
    )
    return OBMLM(cfg)


def lr_at(step, total, peak, warmup=150):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def evaluate_all(mesh, streams, keys, ctx):
    """每域固定窗口的集成 ppl/acc。"""
    out = {}
    for k in keys:
        seq, tgt = streams[k].eval_windows()
        ppls, accs = [], []
        for b in range(0, len(seq), 12):
            p, a = mesh.eval_ppl(seq[b:b + 12], tgt[b:b + 12],
                                 np.ones(ctx, dtype=bool))
            ppls.append(p); accs.append(a)
        out[k] = dict(ppl=float(np.mean(ppls)), acc=float(np.mean(accs)))
    return out


def generate(mesh, streams, tok, keys, ctx=256, n_new=96):
    """每域采样：集成路由贪心解码。"""
    samples = {}
    for k in keys:
        seq, _ = streams[k].eval_windows(n=1, ctx=ctx)
        ids = list(seq[0][:48])
        for _ in range(n_new):
            window = np.asarray([ids[-ctx:]], dtype=np.int64)
            mask = np.ones(len(ids[-ctx:]), dtype=bool)
            ctxd = mesh._context(mesh.cells[0].signature(window))
            rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
            sel, _ = mesh._select(rels, allow_reflex=False)
            ps = []
            for i in sel:
                from obm_lm import backend as bk
                lg = bk.forward(mesh.cells[i], window)
                p = np.exp(lg - lg.max(-1, keepdims=True))
                ps.append((p / p.sum(-1, keepdims=True))[0, -1])
            w = np.exp(mesh.cfg.ensemble_beta * rels[sel])
            w /= w.sum()
            dist = sum(wi * p for wi, p in zip(w, ps))
            ids.append(int(np.argmax(dist)))
        samples[k] = tok.decode(np.asarray(ids))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="pilot", choices=list(SCALES))
    ap.add_argument("--steps", type=int, default=0, help="每域步数（0=阶梯默认）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # 每段步数按时间预算压缩（用户侧超时是硬约束；pilot 300 步/段已集齐
    # 封存/分裂/合同/复现全部机制事件，更大规模只需验证机制保持性而非
    # 从头预训练）：S1 400 / S2 300 / S3 250。串行总时长约 7-8h。
    steps_per_block = args.steps or {"pilot": 300, "S1": 400, "S2": 300,
                                     "S3": 250}[args.scale]
    out_dir = args.out or os.path.join(_OBM_ROOT, "results_scale", f"{args.scale}{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    # 6 段流：4 域 + 后两段复现（build_streams 的 key 顺序即 PLAN 顺序，
    # 复现段用独立的 #repris 流——不与首段共享 rng 状态）
    keys = list(streams.keys())
    assert len(keys) == 6, keys
    uniq = ["en_lit", "zh_classic", "code_py", "en_wiki"]

    mesh = build_mesh(args.scale, args.seed, tok.vocab_size)
    total_steps = steps_per_block * len(keys)
    peak_lr = SCALES[args.scale]["lr"]

    print(f"== OBM-LM scale run: {args.scale} | cells={len(mesh.cells)} "
          f"params={mesh.n_params/1e6:.1f}M | steps/block={steps_per_block} "
          f"total={total_steps}", flush=True)

    if args.resume:
        with open(args.resume, "rb") as f:
            mesh.restore(pickle.load(f))
        print(f"== resumed from {args.resume} @ step {mesh.step}", flush=True)

    metrics = dict(scale=args.scale, seed=args.seed, steps_per_block=steps_per_block,
                   n_params=mesh.n_params, d_model=SCALES[args.scale]["d_model"],
                   n_layers=SCALES[args.scale]["n_layers"],
                   vocab=tok.vocab_size, eval_every=args.eval_every,
                   curves={k: [] for k in uniq}, online=[], events=[],
                   block_of={}, started=time.time())
    t0 = time.time()
    tok_count = 0
    last_ckpt = -1
    ev_cursor = 0

    step = mesh.step
    for bi, key in enumerate(keys):
        st = streams[key]
        blk_t0 = time.time()
        for i in range(steps_per_block):
            seq, tgt, mask = st.batch(mesh.cfg.batch, mesh.cfg.max_len)
            lr = lr_at(step, total_steps, peak_lr)
            mesh.cfg.lr = lr
            mesh.cfg.lr_breach = lr / 3
            mesh.cfg.dream_lr = lr
            d = mesh.run_step(seq, tgt, mask)
            tok_count += seq.size
            step += 1
            if (step % args.eval_every == 0) or (i == steps_per_block - 1):
                ev = evaluate_all(mesh, streams, uniq, mesh.cfg.max_len)
                for k in uniq:
                    metrics["curves"][k].append(dict(step=step, block=bi, **ev[k]))
                el = time.time() - t0
                print(f"  s{step} [blk{bi}:{key}] " + " ".join(
                    f"{k}={ev[k]['ppl']:.1f}" for k in uniq) +
                      f" | online={d['ens_acc']:.3f} cells={len(mesh.cells)} "
                      f"seg={mesh.seg_counter} {el/60:.1f}min", flush=True)
            if step % 25 == 0:
                metrics["online"].append(dict(step=step, block=bi,
                                              acc=d["ens_acc"], k=len(d["selected"])))
            if step - last_ckpt >= args.ckpt_every:
                with open(os.path.join(out_dir, "ckpt_latest.pkl"), "wb") as f:
                    pickle.dump(mesh.checkpoint(), f, protocol=4)
                last_ckpt = step
                print(f"  [ckpt] step {step} saved "
                      f"({time.time()-t0:.0f}s)", flush=True)
        for s, kind, msg in mesh.events[ev_cursor:]:
            metrics["events"].append(dict(step=s, kind=kind, msg=msg,
                                          block=bi))
        ev_cursor = len(mesh.events)
        print(f"== block {bi} ({key}) done in {time.time()-blk_t0:.0f}s, "
              f"cells={len(mesh.cells)}", flush=True)

    # ================= 终局评估 =================
    final = evaluate_all(mesh, streams, uniq, mesh.cfg.max_len)
    metrics["final"] = final
    # best ppl：该域自己的训练段（首现块 + 复现块）全程最低
    best = {}
    for k in uniq:
        best[k] = min(c["ppl"] for c in metrics["curves"][k])
    metrics["best_ppl"] = best
    # forgetting：复现域 final - best（ppl 越低越好，正数=退化）
    metrics["forgetting"] = {k: final[k]["ppl"] - best[k]
                             for k in ("en_lit", "zh_classic")}
    # reprise 恢复：复现段前 100 步的 ppl 均值 vs 专才平台（best）
    reprise = {}
    for k in ("en_lit", "zh_classic"):
        pts = [c["ppl"] for c in metrics["curves"][k] if c["block"] >= 4][:1]
        reprise[k] = pts[0] if pts else None
    metrics["reprise_first_eval"] = reprise
    metrics["cells_final"] = len(mesh.cells)
    metrics["n_params_final"] = mesh.n_params
    metrics["wall_sec"] = time.time() - t0
    metrics["tokens"] = tok_count
    metrics["tok_per_sec"] = tok_count / metrics["wall_sec"]
    metrics["territory"] = [
        dict(cell=c.cell_id,
             seg_rel={str(k): round(v, 3) for k, v in c.seg_rel.items()},
             seg_best={str(k): round(v, 3) for k, v in c.seg_best.items()})
        for c in mesh.cells]
    metrics["events_all"] = [(s, k, m) for s, k, m in mesh.events]

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "ckpt_final.pkl"), "wb") as f:
        pickle.dump(mesh.checkpoint(), f, protocol=4)

    # ================= 采样 =================
    samples = generate(mesh, streams, tok, uniq, ctx=mesh.cfg.max_len)
    with open(os.path.join(out_dir, "samples.txt"), "w") as f:
        for k, s in samples.items():
            f.write(f"===== {k} =====\n{s}\n\n")
    print("== samples saved", flush=True)
    print(json.dumps({k: final[k] for k in final}, ensure_ascii=False), flush=True)
    print(f"== DONE {args.scale}: wall={metrics['wall_sec']/60:.1f}min "
          f"params={metrics['n_params_final']/1e6:.1f}M "
          f"cells={metrics['cells_final']} "
          f"tok/s={metrics['tok_per_sec']:.0f}", flush=True)


if __name__ == "__main__":
    main()
