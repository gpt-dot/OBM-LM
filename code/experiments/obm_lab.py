"""obm_lab.py —— 跳跃型迭代实验台：快速扫配置、计时、量内存、验事件点火。

目标函数（与用户诉求对齐）：在不损失 OBM 功能（封存/分裂/凋亡/复现事件
照 pilot_v1 的点火频率点火）的前提下，压 step_time 与峰值内存。

用法（单次跑一个配置）：
    python3 experiments/obm_lab.py --tag base \
        --d-model 256 --n-layers 4 --n-heads 8 --steps-per-domain 250
    python3 experiments/obm_lab.py --tag half --d-model 192 --n-layers 3 ...
    python3 experiments/obm_lab.py --profile --steps-per-domain 30   # 只看耗时分解

判据（绝不谎报）：
- events 列表非空且含 seal/split/reprise 才认"功能在"；
- step_time_mean 与 pilot_v1（~2.2s）对比看提速；
- peak_rss_mb 看内存。
"""
import os, sys, time, json, argparse, resource, math

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OBM_THREADS", "8")))

from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm import corpus
from obm_lm.tokenizer import BPETokenizer

# 根目录可经环境变量 OBM_ROOT 覆盖（默认本地沙箱路径），便于远端部署
_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_OBM_ROOT, "data", "cache")


def lr_at(step, total, peak, warmup=150):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def build_mesh(a, vocab):
    cfg = OBMLMConfig(
        seed=a.seed, backend="torch", vocab=vocab,
        d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
        max_len=a.ctx, batch=a.batch,
        n_cells_init=a.n_cells, max_cells=a.max_cells,
        out_full=True, acc_prior=0.0, amp_bf16=not a.no_amp,
        grad_clip=1.0, weight_decay=0.01,
        lr=a.lr, lr_breach=a.lr / 3, dream_lr=a.lr,
        fulfill_acc=a.fulfill_acc, expert_acc=a.expert_acc,
        specialist_line=a.specialist_line,
        award_reprise_d=3.0, award_reprise_probe=0.20,
        split_rel_frac=0.85, split_cooldown=400,
        dream_batches=a.dream_batches, dream_batch=a.dream_batch,
        dream_period=a.dream_period,
        buffer_size=600, seg_buffer_size=120,
        dist_refresh=a.dist_refresh,
        gauss_chi=0.0,
    )
    return OBMLM(cfg)


def evaluate_all(mesh, streams, keys, ctx):
    """每域固定窗口的集成 ppl/acc（与 run_scale 同判据，验证效果不变）。"""
    out = {}
    for k in keys:
        seq, tgt = streams[k].eval_windows(ctx=ctx)
        ppls, accs = [], []
        for b in range(0, len(seq), 12):
            p, a = mesh.eval_ppl(seq[b:b + 12], tgt[b:b + 12],
                                 np.ones(ctx, dtype=bool))
            ppls.append(p); accs.append(a)
        out[k] = dict(ppl=float(np.mean(ppls)), acc=float(np.mean(accs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    # 架构
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    # 路由/梦境开销
    ap.add_argument("--dist-refresh", type=int, default=10)
    ap.add_argument("--dream-period", type=int, default=200)
    ap.add_argument("--dream-batches", type=int, default=8)
    ap.add_argument("--dream-batch", type=int, default=16)
    # 记账门限
    ap.add_argument("--fulfill-acc", type=float, default=0.12)
    ap.add_argument("--expert-acc", type=float, default=0.30)
    ap.add_argument("--specialist-line", type=float, default=0.80)
    # 运行
    ap.add_argument("--steps-per-domain", type=int, default=250)
    ap.add_argument("--n-blocks", type=int, default=3)  # 3=A,B,A(reprise)
    ap.add_argument("--domains", default=None,
                    help="逗号分隔的域序列（覆盖 --n-blocks 的自动计划），"
                         "如 code_py,en_wiki 或 en_lit,zh_classic,code_py,en_wiki")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="交错重放概率 p>0：每步以 p 概率把训练 batch 换成"
                         "其他域（全部 DOMAINS 减当前域）的 batch，抗灾难性遗忘")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="lab")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--profile", action="store_true", help="跑 cProfile 只看耗时分解")
    ap.add_argument("--profile-steps", type=int, default=30)
    ap.add_argument("--eval", action="store_true", help="终局按域评估 ppl/acc（验证效果）")
    ap.add_argument("--weights-dir", default=None,
                    help="权重-only 存档目录（默认 results_lab）；设为 scnet efile 路径则远端直接落盘")
    ap.add_argument("--weights-every", type=int, default=0,
                    help="每 N 步存一次权重-only 检查点（0=仅终局）；远端训练建议 1000")
    ap.add_argument("--resume", default=None,
                    help="从权重-only 存档续训（加载参数，优化器从头累积）")
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    keys = list(streams.keys())
    # 迭代用域序列：显式 --domains 优先，否则前 n-blocks 个域
    if args.domains:
        plan = [s.strip() for s in args.domains.split(",") if s.strip()]
        for k in plan:
            assert k in streams, f"unknown domain {k!r} (have {keys})"
    else:
        plan = keys[:args.n_blocks]

    mesh = build_mesh(args, tok.vocab_size)
    if args.resume:
        mesh.load_weights_only(args.resume)
        print(f"  [resume] loaded {args.resume} @step{mesh.step}", flush=True)
    total = args.steps_per_domain * len(plan)
    peak_lr = args.lr
    print(f"== [{args.tag}] cells={len(mesh.cells)} params={mesh.n_params/1e6:.2f}M "
          f"D={args.d_model} L={args.n_layers} B={args.batch} "
          f"dr={args.dist_refresh} dp={args.dream_period} "
          f"steps/blk={args.steps_per_domain} total={total}", flush=True)

    if args.profile:
        import cProfile, pstats, io
        st = streams[plan[0]]
        pr = cProfile.Profile()
        pr.enable()
        step = 0
        for i in range(args.profile_steps):
            seq, tgt, mask = st.batch(mesh.cfg.batch, mesh.cfg.max_len)
            mesh.cfg.lr = peak_lr; mesh.cfg.lr_breach = peak_lr/3; mesh.cfg.dream_lr = peak_lr
            mesh.run_step(seq, tgt, mask)
            step += 1
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(28)
        print(s.getvalue())
        s2 = io.StringIO()
        ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
        ps2.print_stats(20)
        print(s2.getvalue())
        return

    t0 = time.time()
    step_times = []
    tok_count = 0
    ev_cursor = 0
    online = []
    # 交错重放池：全部基础域（重放时随机挑一个非当前域的流）
    _base = corpus.DOMAINS
    _replay_rng = np.random.default_rng(args.seed + 777)
    weights_dir = args.weights_dir or os.path.join(_OBM_ROOT, "results_lab")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{args.tag}_weights.pkl")
    # 远端训练默认存 fp16（文件减半、便于拉回 8GB 沙箱）；设 OBM_WD_FP16=0 强制 fp32
    _fp16 = os.environ.get("OBM_WD_FP16", "1") not in ("0", "false", "False")
    wd_dtype = np.float16 if _fp16 else np.float32
    print(f"  [weights] dtype={'fp16' if _fp16 else 'fp32'} -> {weights_path}", flush=True)
    step = mesh.step
    start_step = step  # resume 时 lr 调度以本次会话进度为基准
    n_replayed = 0
    for bi, key in enumerate(plan):
        st = streams[key]
        for i in range(args.steps_per_domain):
            seq, tgt, mask = st.batch(mesh.cfg.batch, mesh.cfg.max_len)
            if args.replay > 0 and _replay_rng.random() < args.replay:
                # 交错重放：换成随机其他域的 batch（旧知识保温）
                others = [k for k in _base if k != key]
                rk = others[_replay_rng.integers(0, len(others))]
                seq, tgt, mask = streams[rk].batch(mesh.cfg.batch, mesh.cfg.max_len)
                n_replayed += 1
            lr = lr_at(step - start_step, total, peak_lr)
            mesh.cfg.lr = lr; mesh.cfg.lr_breach = lr/3; mesh.cfg.dream_lr = lr
            ts = time.time()
            d = mesh.run_step(seq, tgt, mask)
            step_times.append(time.time() - ts)
            tok_count += seq.size
            step += 1
            if step % 25 == 0:
                online.append(dict(step=step, block=bi, acc=d["ens_acc"],
                                   k=len(d["selected"]),
                                   peak=round(mesh.peak_acc, 3),
                                   cells=len(mesh.cells)))
            if args.weights_every and step % args.weights_every == 0:
                mesh.save_weights_only(weights_path, wd_dtype)
        print(f"  [blk{bi}:{key}] done @step{step} cells={len(mesh.cells)} "
              f"seg={mesh.seg_counter} replayed={n_replayed} {time.time()-t0:.0f}s", flush=True)
    # 终局权重-only 存档（沙箱推理用）
    mesh.save_weights_only(weights_path, wd_dtype)
    print(f"  [weights] saved -> {weights_path} "
          f"({mesh.n_params/1e6:.1f}M params)", flush=True)

    wall = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    step_times = np.array(step_times)
    # 事件统计
    kinds = {}
    for s, kind, msg in mesh.events:
        kinds[kind] = kinds.get(kind, 0) + 1
    events = [(s, k, m[:80]) for s, k, m in mesh.events]
    out = dict(tag=args.tag, d_model=args.d_model, n_layers=args.n_layers,
               batch=args.batch, ctx=args.ctx, dist_refresh=args.dist_refresh,
               dream_period=args.dream_period, amp=not args.no_amp,
               n_params=mesh.n_params, cells_final=len(mesh.cells),
               total_steps=step, wall_sec=wall,
               step_time_mean=float(step_times.mean()),
               step_time_p50=float(np.median(step_times)),
               step_time_p90=float(np.percentile(step_times, 90)),
               tok_per_sec=tok_count / wall,
               peak_rss_mb=peak_mb, event_counts=kinds, events=events,
               segments=mesh.seg_counter, online=online,
               territory=[
                   dict(cell=c.cell_id,
                        seg_rel={str(k): round(v, 3) for k, v in c.seg_rel.items()},
                        seg_best={str(k): round(v, 3) for k, v in c.seg_best.items()})
                   for c in mesh.cells],
               peak_acc_final=round(mesh.peak_acc, 4))
    if args.eval:
        uniq = ["en_lit", "zh_classic", "code_py", "en_wiki"]
        ev = evaluate_all(mesh, streams, uniq, mesh.cfg.max_len)
        out["final_ppl"] = {k: ev[k]["ppl"] for k in uniq}
        out["final_acc"] = {k: ev[k]["acc"] for k in uniq}
        print("  [eval] " + " ".join(f"{k}={ev[k]['ppl']:.0f}" for k in uniq),
              flush=True)
    os.makedirs(os.path.join(_OBM_ROOT, "results_lab"), exist_ok=True)
    # 功能判据：seal=边界检测底线；birth=分裂；death=凋亡；forget=记忆管理
    kc = out["event_counts"]
    verdict = {
        "seal(封存/边界)": kc.get("seal", 0) > 0,
        "birth(分裂)": kc.get("birth", 0) > 0,
        "death(凋亡)": kc.get("death", 0) > 0,
    }
    out["function_verdict"] = verdict
    out["function_pass"] = verdict["seal(封存/边界)"]   # 底线：封存必须点火
    with open(os.path.join(_OBM_ROOT, "results_lab", f"{args.tag}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    fv = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in verdict.items())
    print(f"== [{args.tag}] DONE wall={wall/60:.1f}min "
          f"step={out['step_time_mean']*1000:.0f}ms(p50 {out['step_time_p50']*1000:.0f}ms) "
          f"tok/s={out['tok_per_sec']:.0f} peakRSS={peak_mb:.0f}MB "
          f"params={out['n_params']/1e6:.2f}M cells={out['cells_final']} "
          f"events={kinds} | 功能 {fv} pass={out['function_pass']}", flush=True)


if __name__ == "__main__":
    main()
