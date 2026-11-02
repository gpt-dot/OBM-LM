"""run_scale_fast.py —— 共享 trunk + 薄专才 head 架构的训练验证。

对照原 OBM-LM（N 个全量 transformer 细胞），验证：
  * 同性质：概念签名 / 条件合同市场 / 双信号边界 / 单份+复现授予 / 梦境代谢
    全部保留（mesh 零改动，仅细胞身体换成 trunk+head）；
  * 低算力、快进度：每步只 1 次 trunk 前向（冻结）+ 1 次薄 head 前向/反向，
    不再有 N× 全量前向；
  * 快 tok'：推理 1 次 trunk + k 次薄 head，替代原 N× 全量前向；
  * 规模不变：trunk 可做到 LLM 量级（d=768/L=10）。

用法：
  python3 experiments/run_scale_fast.py --scale S1 --head-layers 1 \
      --trunk-layers 6 --pretrain 400 --steps 200 --tag _v1 --seed 0
"""
import os, sys, time, json, math, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from obm_lm.tokenizer import BPETokenizer
from obm_lm import corpus
from obm_lm.fast_cell import OBMLM_Fast
from run_scale import SCALES, lr_at

CACHE = os.path.join(_ROOT, "obm_mvp/data/cache")
OUT_BASE = os.path.join(_ROOT, "obm_mvp/results_scale")
UNIQ = ["en_lit", "zh_classic", "code_py", "en_wiki"]


def build_fast_cfg(scale, seed, vocab, head_layers, trunk_layers):
    s = SCALES[scale]
    from obm_lm.lm_mesh import OBMLMConfig
    cfg = OBMLMConfig(
        seed=seed, backend="torch", vocab=vocab,
        d_model=s["d_model"], n_heads=s["n_heads"], n_layers=trunk_layers,
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
        gauss_chi=0.0,
    )
    return cfg


def pretrain_trunk(mesh, streams, steps, lr=1e-3, trunk_lr=0.0):
    """trunk 短预训练。混合语料上做标准 LM。

    trunk_lr>0 时：trunk 在 OBM 阶段继续以极小 lr 随梯度进化（共享、每步只算 1
    次，速度优势保住），返回 AdamW 优化器供主循环每步 step；否则训练后冻结。
    """
    import torch
    opt = torch.optim.AdamW(mesh.trunk.parameters(), lr=lr, weight_decay=0.01)
    keys = list(streams.keys())
    t0 = time.time()
    for stp in range(steps):
        key = keys[stp % len(keys)]
        seq, tgt, _ = streams[key].batch(mesh.cfg.batch, mesh.cfg.max_len)
        seq = torch.as_tensor(np.asarray(seq), dtype=torch.long)
        tgt = torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long)
        opt.zero_grad()
        h = mesh.trunk(seq, train=True)
        B, L, D = h.shape
        logits = (torch.nn.functional.layer_norm(h, (D,)) @ mesh.trunk.tok_emb.t()).reshape(-1, mesh.trunk.V)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        opt.step()
        if (stp + 1) % 50 == 0:
            print(f"  [pretrain] step {stp+1}/{steps} loss={loss.item():.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if trunk_lr and trunk_lr > 0:
        # 不冻结：返回专用优化器，主循环每步 step（极小 lr 联合训练共享主干）
        topt = torch.optim.AdamW(mesh.trunk.parameters(), lr=trunk_lr,
                                 weight_decay=0.01)
        mesh.set_trunk_train(True)
        print(f"== trunk pretrain done ({steps} steps, {time.time()-t0:.0f}s), "
              f"KEEP TRAINABLE (trunk-lr={trunk_lr}) for OBM phase", flush=True)
        return topt
    # 冻结 trunk：OBM 阶段不再更新
    for p in mesh.trunk.parameters():
        p.requires_grad_(False)
    print(f"== trunk pretrain done ({steps} steps, {time.time()-t0:.0f}s), "
          f"frozen for OBM phase", flush=True)
    return None


def evaluate_all(mesh, streams, keys, ctx):
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
    samples = {}
    for k in keys:
        seq, _ = streams[k].eval_windows(n=1, ctx=ctx)
        ids = list(seq[0][:48])
        for _ in range(n_new):
            window = np.asarray([ids[-ctx:]], dtype=np.int64)
            sig = mesh.cells[0].signature(window)
            ctxd = mesh._context(sig)
            rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
            sel, _ = mesh._select(rels, allow_reflex=False)
            ps = []
            for i in sel:
                from obm_lm import backend as bk
                lg = bk.forward(mesh.cells[i], window)
                if hasattr(lg, "detach"):          # torch 张量 → numpy
                    lg = lg.detach().float().cpu().numpy()
                lg = lg - lg.max(-1, keepdims=True)  # 数值稳定
                p = np.exp(lg)
                ps.append((p / p.sum(-1, keepdims=True))[0, -1])
            w = np.exp(mesh.cfg.ensemble_beta * rels[sel]); w /= w.sum()
            dist = sum(wi * p for wi, p in zip(w, ps))
            ids.append(int(np.argmax(dist)))
        samples[k] = tok.decode(np.asarray(ids))
    return samples


def _dump(metrics, out_dir):
    """增量落盘：去掉私有键，补可见派生字段（被杀也能留数据）。"""
    m = {k: v for k, v in metrics.items() if not k.startswith("_")}
    if m.get("curves"):
        m["best_ppl"] = {k: min(c["ppl"] for c in m["curves"][k])
                         for k in UNIQ if m["curves"].get(k)}
        m["n_params"] = (m.get("n_params_final") or m.get("n_params"))
    if m.get("step_times"):
        st = m["step_times"]
        m["step_time_mean"] = float(np.mean(st))
        m["step_time_p50"] = float(np.percentile(st, 50))
    m.pop("_last_ckpt", None)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="S1", choices=list(SCALES))
    ap.add_argument("--steps", type=int, default=200, help="每域步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_v1")
    ap.add_argument("--trunk-layers", type=int, default=None,
                   help="trunk 层数（默认=SCALES 的 n_layers）")
    ap.add_argument("--head-layers", type=int, default=1, help="薄 head 层数")
    ap.add_argument("--pretrain", type=int, default=400, help="trunk 预训练步数(0=跳过)")
    ap.add_argument("--trunk-lr", type=float, default=0.0,
                   help=">0 时 trunk 在 OBM 阶段以极小 lr 联合训练（共享，每步1次）")
    ap.add_argument("--resume", default=None, help="ckpt 路径，断点续训")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=400)
    args = ap.parse_args()

    steps_per_block = args.steps
    out_dir = os.path.join(OUT_BASE, f"fast{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    streams = corpus.build_streams(CACHE)
    keys = list(streams.keys())
    assert len(keys) == 6, keys

    cfg = build_fast_cfg(args.scale, args.seed, tok.vocab_size,
                         args.head_layers, args.trunk_layers)
    total_steps = steps_per_block * len(keys)
    trunk_train = args.trunk_lr > 0

    resume = bool(args.resume) and os.path.exists(args.resume)
    if resume:
        import pickle
        ck = pickle.load(open(args.resume, "rb"))
        mesh = OBMLM_Fast(cfg, trunk_layers=args.trunk_layers,
                         head_layers=args.head_layers, trunk_train=trunk_train)
        mesh.restore(ck)
        mp = os.path.join(out_dir, "metrics.json")
        metrics = json.load(open(mp)) if os.path.exists(mp) else None
        if metrics is None:
            metrics = dict(curves={k: [] for k in UNIQ}, online=[], events=[],
                           block_of={})
        done_blocks = int(metrics.get("done_blocks", 0))
        step = int(metrics.get("step", done_blocks * steps_per_block))
        step_times = list(metrics.get("step_times", []))
        prev_wall = float(metrics.get("prev_wall_sec", 0.0))
        ev_cursor = len(mesh.events)
        print(f"== RESUME from {args.resume}: done_blocks={done_blocks} "
              f"step={step} cells={len(mesh.cells)}", flush=True)
    else:
        mesh = OBMLM_Fast(cfg, trunk_layers=args.trunk_layers,
                         head_layers=args.head_layers, trunk_train=trunk_train)
        done_blocks, step, step_times, prev_wall, ev_cursor = 0, 0, [], 0.0, 0
        metrics = dict(scale="fast", seed=args.seed, steps_per_block=steps_per_block,
                       trunk_layers=args.trunk_layers, head_layers=args.head_layers,
                       trunk_lr=args.trunk_lr,
                       n_params=mesh.trunk.n_params() + sum(c.n_params() for c in mesh.cells),
                       d_model=cfg.d_model, n_layers=cfg.n_layers, vocab=tok.vocab_size,
                       eval_every=args.eval_every, curves={k: [] for k in UNIQ},
                       online=[], events=[], block_of={}, started=time.time())

    print(f"== OBM-LM-FAST {args.scale} | trunk={cfg.d_model}x{args.trunk_layers} "
          f"head={args.head_layers}L | cells={len(mesh.cells)} "
          f"trunk_params={mesh.trunk.n_params()/1e6:.1f}M "
          f"head_params={sum(c.n_params() for c in mesh.cells)/1e6:.2f}M "
          f"trunk-lr={args.trunk_lr} steps/block={steps_per_block} total={total_steps}",
          flush=True)

    # trunk 优化器（仅 trunk-lr>0 时存在；每步随 OBM 联合训练共享主干）
    trunk_opt = None
    if args.trunk_lr > 0:
        import torch
        if args.pretrain > 0:
            trunk_opt = pretrain_trunk(mesh, streams, args.pretrain,
                                       trunk_lr=args.trunk_lr)
        else:
            for p in mesh.trunk.parameters():
                p.requires_grad_(True)
            mesh.set_trunk_train(True)
            trunk_opt = torch.optim.AdamW(mesh.trunk.parameters(),
                                          lr=args.trunk_lr, weight_decay=0.01)
    elif args.pretrain > 0:
        pretrain_trunk(mesh, streams, args.pretrain, trunk_lr=0.0)

    t0 = time.time()
    for bi in range(done_blocks, len(keys)):
        key = keys[bi]
        blk_t0 = time.time()
        metrics["done_blocks"] = bi
        for i in range(steps_per_block):
            seq, tgt, mask = streams[key].batch(cfg.batch, cfg.max_len)
            lr = lr_at(step, total_steps, SCALES[args.scale]["lr"])
            mesh.cfg.lr = lr
            mesh.cfg.lr_breach = lr / 3
            mesh.cfg.dream_lr = lr
            ts = time.time()
            d = mesh.run_step(seq, tgt, mask)
            step_times.append(time.time() - ts)
            step += 1
            if trunk_opt is not None:
                trunk_opt.step()
                trunk_opt.zero_grad()
            if (step % args.eval_every == 0) or (i == steps_per_block - 1):
                ev = evaluate_all(mesh, streams, UNIQ, cfg.max_len)
                for k in UNIQ:
                    metrics["curves"][k].append(dict(step=step, block=bi, **ev[k]))
                el = time.time() - t0 + prev_wall
                print(f"  s{step} [blk{bi}:{key}] " + " ".join(
                    f"{k}={ev[k]['ppl']:.1f}" for k in UNIQ) +
                      f" | online={d['ens_acc']:.3f} cells={len(mesh.cells)} "
                      f"seg={mesh.seg_counter} {el/60:.1f}min", flush=True)
            if step % 25 == 0:
                metrics["online"].append(dict(step=step, block=bi,
                                              acc=d["ens_acc"], k=len(d["selected"])))
            metrics["step"] = step
            metrics["step_times"] = step_times
            metrics["prev_wall_sec"] = time.time() - t0
            if step - metrics.get("_last_ckpt", -999) >= args.ckpt_every:
                import pickle
                with open(os.path.join(out_dir, "ckpt_latest.pkl"), "wb") as f:
                    pickle.dump(mesh.checkpoint(), f, protocol=4)
                metrics["_last_ckpt"] = step
                _dump(metrics, out_dir)   # 增量落盘
                print(f"  [ckpt] step {step} saved ({time.time()-t0:.0f}s)", flush=True)
        for s, kind, msg in mesh.events[ev_cursor:]:
            metrics["events"].append(dict(step=s, kind=kind, msg=msg, block=bi))
        ev_cursor = len(mesh.events)
        metrics["block_of"][key] = bi
        metrics["done_blocks"] = bi + 1
        _dump(metrics, out_dir)
        print(f"== block {bi} ({key}) done in {time.time()-blk_t0:.0f}s, "
              f"cells={len(mesh.cells)}", flush=True)

    final = evaluate_all(mesh, streams, UNIQ, cfg.max_len)
    metrics["final"] = final
    metrics["best_ppl"] = {k: min(c["ppl"] for c in metrics["curves"][k])
                           for k in UNIQ}
    metrics["forgetting"] = {k: final[k]["ppl"] - metrics["best_ppl"][k]
                             for k in ("en_lit", "zh_classic")}
    metrics["cells_final"] = len(mesh.cells)
    metrics["n_params_final"] = mesh.trunk.n_params() + sum(c.n_params() for c in mesh.cells)
    metrics["wall_sec"] = time.time() - t0 + prev_wall
    metrics["tokens"] = total_steps * cfg.batch * cfg.max_len
    metrics["tok_per_sec"] = metrics["tokens"] / max(1e-6, metrics["wall_sec"])
    metrics["step_time_mean"] = float(np.mean(step_times))
    metrics["step_time_p50"] = float(np.percentile(step_times, 50))
    metrics["territory"] = [dict(cell=c.cell_id,
                                seg_rel={str(k): round(v, 3) for k, v in c.seg_rel.items()},
                                seg_best={str(k): round(v, 3) for k, v in c.seg_best.items()})
                            for c in mesh.cells]
    metrics["events_all"] = [(s, k, m) for s, k, m in mesh.events]
    _dump(metrics, out_dir)
    with open(os.path.join(out_dir, "ckpt_final.pkl"), "wb") as f:
        import pickle
        pickle.dump(mesh.checkpoint(), f, protocol=4)
    samples = generate(mesh, streams, tok, UNIQ, ctx=cfg.max_len)
    with open(os.path.join(out_dir, "samples.txt"), "w") as f:
        for k, s in samples.items():
            f.write(f"===== {k} =====\n{s}\n\n")
    print(f"== done. step_time(mean/p50)={metrics['step_time_mean']:.3f}/"
          f"{metrics['step_time_p50']:.3f}s | wall={metrics['wall_sec']/60:.1f}min "
          f"| cells={metrics['cells_final']} | events={len(metrics['events_all'])}",
          flush=True)


if __name__ == "__main__":
    main()
