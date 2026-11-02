"""backend.py —— OBM 张量引擎分派层：numpy mini-autograd ↔ PyTorch。

mesh（lm_mesh.py）的全部训练/推理调用收敛到这几个函数，引擎可换：
    make_cell(...)   建细胞（按 cfg.backend 选 MicroLM / TorchMicroLM）
    forward(c, idx, noise, train)  logits：train=False -> numpy (B,L,V)；
                                  train=True -> 带 grad 的张量（numpy 版
                                  返回 autograd T，torch 版返回 Tensor）
    npv(t)           取 logits 的 numpy 视图（.data / .detach().numpy()）
    ce_step(mesh, c, logits, tgt, lr)   zero_grad+CE+backward+(clip)+step，
                                        返回 loss 标量（numpy/torch 双实现，
                                        Adam 语义对齐：bias correction、
                                        (0.9,0.999,1e-8)）

为什么留着 numpy 路：MVP/玩具实验（run_lm.py、diag_*.py）的已验证轨迹
靠它复现；规模实验（run_scale.py）走 torch 路。两条路共用同一 mesh 代码，
是"换引擎不动灵魂"的落实。
"""
import numpy as np

_TORCH = None      # lazily imported


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def is_torch(cell):
    return getattr(cell, "backend", None) == "torch"


def make_cell(rng, vocab, d_model, n_heads, n_layers, max_len, cell_id,
              sig_emb, cfg=None):
    if cfg is not None and getattr(cfg, "backend", "numpy") == "torch":
        from obm_lm.torch_cell import TorchMicroLM
        return TorchMicroLM(rng, vocab, d_model, n_heads, n_layers, max_len,
                            cell_id, sig_emb, cfg)
    from obm_lm.cell_lm import MicroLM
    return MicroLM(rng, vocab, d_model, n_heads, n_layers, max_len, cell_id,
                   sig_emb)


def forward(cell, idx, noise=0.0, train=False):
    """idx: numpy (B,L) int -> logits。

    train=False: numpy (B,L,V)（无 grad；numpy 版即 T.data）
    train=True : 带 grad 的张量（numpy 版为 autograd T，torch 版为 Tensor）
    """
    if is_torch(cell):
        return cell.forward_torch(idx, noise=noise, train=train)
    t = cell.forward(idx, noise=noise)
    return t if train else t.data


def npv(t):
    """logits 张量的 numpy 视图。

    numpy autograd T -> .data（numpy）；torch.Tensor -> .detach().numpy()；
    已是 numpy 则原样返回。注意不能用 isinstance(t, T)——torch.Tensor 的
    .data 仍是 Tensor，不是 numpy 视图。
    """
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "detach"):                    # torch.Tensor
        return t.detach().float().numpy() if t.dtype == _torch().bfloat16 \
            else t.detach().numpy()
    return t.data                               # numpy autograd T


def forward_stats(cell, idx, mask=None, tgt=None):
    """路由统计（dist/top1/probe）。torch 路免物化（V=16K 时 (B,L,V) 的
    numpy softmax 是 400MB/细胞/步，实测拖慢 4 倍）；numpy 路 V=16 直接算。"""
    if is_torch(cell):
        return cell.forward_stats(idx, mask, tgt)
    lg = cell.forward(idx).data                 # (B,L,V) numpy
    p = np.exp(lg - lg.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    L = lg.shape[1]
    m = np.ones(L, dtype=bool) if mask is None else np.asarray(mask)
    out = dict(dist=p.mean(axis=(0, 1)), top1=float(np.mean(p.max(axis=-1)[:, m])))
    if tgt is not None:
        pred = np.argmax(lg, -1)
        out["probe"] = float(np.mean(pred[:, m] == tgt[:, m]))
    return out


def cell_state(cell):
    """细胞 checkpoint 状态（引擎分派：参数 + Adam 矩 + 合同账本）。"""
    return cell.state()


def cell_load(cell, st):
    """把 checkpoint 状态载入细胞（就地替换参数，恢复 Adam 矩）。"""
    cell.load_state(st)


def top1_acc(cell, lg, mask, tgt):
    """单细胞输出段 next-token 精度。torch 路 argmax 后只搬 (B,L) 回
    numpy——直接 npv(lg) 是 (B,L,V)=402MB/细胞/步的物化。"""
    if is_torch(cell):
        with _torch().no_grad():
            pred = lg.argmax(-1).numpy()
    else:
        pred = np.argmax(npv(lg), -1)
    return float(np.mean(pred[:, mask] == tgt[:, mask]))


def _ens_torch(cells, sel, w, seq, lg_list=None):
    """torch 路集成分布（no_grad，全程不落地 numpy）。lg_list 给出已建好的
    logits（decide 的训练路径复用带 grad 前向，不重算）。"""
    torch = _torch()
    ens = None
    for k, i in enumerate(sel):
        lg = lg_list[k] if lg_list is not None else cells[i].logits_torch(seq)
        p = torch.softmax(lg.float(), dim=-1)
        wi = float(w[k])
        ens = wi * p if ens is None else ens + wi * p
    return ens


def ens_acc(cells, sel, w, seq, tgt, mask, lg_list=None):
    """集成输出段精度（mask 位置的 next-token 精度）。

    torch 路全程在 torch 内算——旧实现把每个细胞的 (B,L,V) softmax 搬进
    numpy 再 sum(wi*p)，V=16K 时 402MB/细胞/步，decide 实测 1.6s/步。
    numpy 路 V=16 直接算（玩具轨迹不变）。
    """
    if is_torch(cells[0]):
        torch = _torch()
        with torch.no_grad():
            ens = _ens_torch(cells, sel, w, seq, lg_list)
            pred = ens.argmax(-1).numpy()
        return float(np.mean(pred[:, mask] == tgt[:, mask]))
    ps = []
    for k, i in enumerate(sel):
        lg = lg_list[k] if lg_list is not None else forward(cells[i], seq)
        lgn = npv(lg)
        p = np.exp(lgn - lgn.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        ps.append(p)
    ens = sum(float(wi) * p for wi, p in zip(w, ps))
    return float(np.mean(np.argmax(ens, -1)[:, mask] == tgt[:, mask]))


def decide_stats(cells, sel, w, lg_list, tgt, mask):
    """融合 decide 的两处 V 维统计（per-cell top1_acc + ensemble acc）。

    数学事实（省一次完整 V 维遍历的关键）：single-grant + 任期锁下，
    selected 通常只有 1 个细胞，集成权重 w=[1.0]，ensemble 分布就是
    该细胞的 softmax——softmax 单调 ⇒ ensemble argmax ≡ per-cell argmax。
    此时直接用 per-cell argmax 复用，省掉 ens_acc 里那次 fp32 softmax
    物化（(B,L,V)=402MB）+ 一次 V 维 argmax。语义完全等价，零功能风险。

    多细胞路径与原 ens_acc 逐行同果（per-cell argmax + 加权 softmax 累加），
    只是把 top1_acc 的 N 次 argmax 与 ensemble 的遍历合到同一 no_grad 块。
    """
    if not is_torch(cells[0]):
        accs = [top1_acc(cells[i], lg, mask, tgt)
                for i, lg in zip(sel, lg_list)]
        ea = ens_acc(cells, sel, w, None, tgt, mask, lg_list=lg_list)
        return accs, ea
    torch = _torch()
    with torch.no_grad():
        m = np.asarray(mask, dtype=bool)
        t = np.asarray(tgt)
        cell_pred = [lg.argmax(-1).numpy() for lg in lg_list]     # 一趟/细胞
        accs = [float(np.mean(cp[:, m] == t[:, m])) for cp in cell_pred]
        if len(sel) == 1:
            ea = accs[0]                    # 数学恒等：单细胞 ensemble≡per-cell
        else:
            ens = None
            for k in range(len(sel)):
                p = torch.softmax(lg_list[k].float(), dim=-1)
                wi = float(w[k])
                ens = wi * p if ens is None else ens + wi * p
            ens_pred = ens.argmax(-1).numpy()
            ea = float(np.mean(ens_pred[:, m] == t[:, m]))
    return accs, ea


def ens_nll(cells, sel, w, seq, tgt):
    """集成困惑度统计 -> (nll_sum, n_tok, n_correct)，全位置计分
    （与 eval_ppl 旧语义一致：clamp 1e-12，mask 不参与）。"""
    if is_torch(cells[0]):
        torch = _torch()
        with torch.no_grad():
            ens = _ens_torch(cells, sel, w, seq)
            B, L, V = ens.shape
            flat_t = torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long)
            p_true = ens.reshape(B * L, V)[torch.arange(B * L), flat_t]
            nll = float(-torch.log(p_true.clamp_min(1e-12)).sum())
            ok = int((ens.argmax(-1) == torch.as_tensor(np.asarray(tgt))).sum())
        return nll, B * L, ok
    ens = None
    for k, i in enumerate(sel):
        lg = forward(cells[i], seq)
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        wi = float(w[k])
        ens = wi * p if ens is None else ens + wi * p
    rows = np.arange(ens.shape[0])[:, None]
    p_true = ens[rows, np.arange(ens.shape[1]), tgt]
    return (float(-np.log(np.maximum(p_true, 1e-12)).sum()),
            int(ens.shape[0] * ens.shape[1]),
            int(np.sum(np.argmax(ens, -1) == tgt)))


def argmax_pred(cell, seq):
    """next-token 预测 (B,L) numpy。torch 路 argmax 后只搬 (B,L) 回——
    直接 forward(train=False) 是 (B,L,V) 全量物化（梦境合同核对用）。"""
    if is_torch(cell):
        with _torch().no_grad():
            return cell.logits_torch(seq).argmax(-1).numpy()
    return np.argmax(cell.forward(seq).data, -1)


def mean_nll(cell, seq, tgt):
    """全位置平均 NLL（分裂选型用：当前段上损失最低的细胞）。
    torch 路 gather 后只回传标量。"""
    if is_torch(cell):
        torch = _torch()
        with torch.no_grad():
            lg = cell.logits_torch(seq)
            B, L, V = lg.shape
            flat = lg.reshape(B * L, V).float()
            t = torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long)
            lp = torch.log_softmax(flat, dim=-1)
            return float(-lp[torch.arange(B * L), t].mean())
    lg = cell.forward(seq).data
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return float(-np.mean(np.log(p.reshape(-1, p.shape[-1])[
        np.arange(tgt.size), tgt.reshape(-1)] + 1e-12)))


def ce_step(mesh, cell, logits, tgt, lr):
    """对已建图的 logits 做：zero_grad → CE(mean) → backward → (clip) → step。
    返回 loss 标量（float）。numpy 路与 torch 路的 Adam 超参一致。"""
    if is_torch(cell):
        torch = _torch()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long))
        loss.backward()
        cell.adam_step(lr, getattr(mesh.cfg, "grad_clip", 0.0))
        return float(loss.detach())
    from obm_lm.autograd import cross_entropy, backward, adam, zero_grad
    ps = cell.param_list()
    zero_grad(ps)
    ce = cross_entropy(logits, tgt)
    backward(ce)
    adam(ps, lr, mesh._opt_state(cell))
    return float(ce.data)
