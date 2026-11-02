"""torch_cell.py —— OBM 语言模型细胞的 PyTorch 后端（LLM 规模用）。

与 numpy 版 MicroLM（cell_lm.py）**接口逐行同构**，mesh 逻辑零改动复用：
    forward(idx, noise, train) / signature(idx) / param_list() / n_params()
    spawn(rng, noise, new_id, birth_step)
    + 合同状态 seg_rel/seg_note/seg_best/birth_step
    + note/erode/freeze/niche/cond_reliability/global_reliability

结构（与 numpy 版同）：pre-norm GPT，权重共享（logits = h @ tok_emb.T）：
    tok_emb + pos_emb → L×[LN → H 头因果自注意力 → 残差 ; LN → FFN(4x) → 残差]
    → LN → logits

与 numpy 版的差异仅在引擎：
- 参数为 nn.Parameter；优化器是细胞自持的 AdamW（mesh 每步可改 lr：
  履约/违约/梦境三种学习率走同一优化器，step 前覆写 param_group["lr"]）
- forward(train=True) 返回带 grad 的 torch.Tensor；train=False 走
  no_grad 返回 numpy——mesh 里所有 `.data` 用法经 backend.forward 统一
- 概念签名 sig_emb 保持**冻结 numpy 数组**（mesh 级共享，不进 state_dict
  的 trainable 集合；checkpoint 由 mesh 另行保存）
- 初始化分布与 numpy 版逐参数一致（同一 numpy rng、同方差），换后端
  不改变权重的先验几何

为什么必须换后端：numpy mini-autograd 在 D=64/L=2 下尚可（~69K 参数），
LLM 规模（D≥256、L≥6、词表≥8K、总参数量 30M+）下 Python 级 autograd 的
逐 op 开销比 torch 的融合 kernel 慢 2-3 个数量级，且无 mkldnn/线程池。
mesh 的"灵魂"（合同市场/段/授予/梦境/复现）是纯 Python+numpy 的记账
逻辑，与张量引擎正交——换引擎不动灵魂。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchMicroLM(nn.Module):
    """一个语言模型细胞（PyTorch 后端）。backend="torch" 标记供 mesh 分派。"""

    backend = "torch"

    def __init__(self, rng, vocab, d_model=64, n_heads=4, n_layers=2,
                 max_len=32, cell_id=0, sig_emb=None, cfg=None):
        super().__init__()
        self.cell_id = cell_id
        self.rng = rng                # 反事实噪声/分裂变异用（numpy，跨后端一致）
        self.V, self.D, self.H, self.L_max = vocab, d_model, n_heads, max_len
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.cfg = cfg
        # 概念签名投影：冻结 numpy（mesh 级共享；单独使用时自建）
        if sig_emb is None:
            sig_emb = rng.normal(0.0, 1.0, (vocab, d_model))
        self.sig_emb = sig_emb
        s = 1.0 / np.sqrt(d_model)
        # 参数容器（与 numpy 版同名同形，初始化分布一致）
        def p(name, shape, scale):
            setattr(self, name, nn.Parameter(
                torch.from_numpy(rng.normal(0.0, scale, shape).astype(np.float32))))
        self.tok_emb = nn.Parameter(
            torch.from_numpy(rng.normal(0.0, 1.0, (vocab, d_model)).astype(np.float32)))
        self.pos_emb = nn.Parameter(
            torch.from_numpy(rng.normal(0.0, 0.02, (max_len, d_model)).astype(np.float32)))
        for l in range(n_layers):
            p(f"l{l}_wqkv", (d_model, 3 * d_model), s)
            setattr(self, f"l{l}_bqkv", nn.Parameter(torch.zeros(3 * d_model)))
            p(f"l{l}_wo", (d_model, d_model), s)
            setattr(self, f"l{l}_bo", nn.Parameter(torch.zeros(d_model)))
            p(f"l{l}_w1", (d_model, 2 * d_model), s)
            setattr(self, f"l{l}_b1", nn.Parameter(torch.zeros(2 * d_model)))
            p(f"l{l}_w2", (2 * d_model, d_model), s)
            setattr(self, f"l{l}_b2", nn.Parameter(torch.zeros(d_model)))
        self.birth_step = 0
        # 条件合同（与 numpy MicroCell 逐行同构）
        self.seg_rel = {}
        self.seg_note = {}
        self.seg_best = {}
        self.activations = 0
        self.correct = 0
        self._opt = None               # AdamW，lazily built（参数增删后重建）
        self._causal = torch.triu(
            torch.full((max_len, max_len), -1e9), diagonal=1)

    # ---------- 参数 ----------
    def param_list(self):
        return list(self.parameters())

    def n_params(self):
        return sum(q.numel() for q in self.parameters())

    def _optimizer(self):
        """细胞自持 AdamW。lr 由 mesh 每步覆写（履约/违约/梦境三速率）。"""
        if self._opt is None:
            wd = float(getattr(self.cfg, "weight_decay", 0.0) or 0.0) \
                if self.cfg is not None else 0.0
            self._opt = torch.optim.AdamW(self.parameters(), lr=1e-3,
                                          betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=wd)
        return self._opt

    def adam_step(self, lr, clip=0.0):
        opt = self._optimizer()
        for g in opt.param_groups:
            g["lr"] = float(lr)
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    def spawn(self, rng, noise, new_id, birth_step):
        """分裂：复制父代参数 + 噪声变异（与 numpy 版同款；优化器状态不继承——
        子细胞是独立个体，从头积累二阶矩）。"""
        child = TorchMicroLM.__new__(TorchMicroLM)
        nn.Module.__init__(child)
        child.cell_id = new_id
        child.rng = self.rng
        child.V, child.D, child.H = self.V, self.D, self.H
        child.n_layers, child.d_head, child.L_max = self.n_layers, self.d_head, self.L_max
        child.cfg = self.cfg
        child.sig_emb = self.sig_emb    # 冻结投影：共享同一份（不复制）
        child.birth_step = birth_step
        for name, q in self.named_parameters():
            noisy = q.data + torch.from_numpy(
                rng.normal(0.0, noise, q.data.shape).astype(np.float32))
            setattr(child, name, nn.Parameter(noisy))
        child.seg_rel = {}
        child.seg_note = {}
        child.seg_best = {}
        child.activations = 0
        child.correct = 0
        child._opt = None
        child._causal = self._causal
        return child

    # ---------- 前向 ----------
    def forward_torch(self, idx, noise=0.0, train=False):
        """idx: numpy (B,L) int -> logits。train=True 带 grad（torch.Tensor），
        False 走 no_grad 返回 numpy (B,L,V)（对齐 numpy 版 .data 语义）。

        cfg.amp_bf16=True 时矩阵乘走 bf16 autocast（CPU AVX512-BF16，
        实测 ~1.7x；参数与 Adam 状态保持 fp32，只激活值降精度——
        与 numpy 路对齐的取值路径统一转回 fp32）。
        """
        amp = bool(getattr(self.cfg, "amp_bf16", False)) if self.cfg is not None else False
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            if amp:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = self._fwd(idx, noise)
            else:
                out = self._fwd(idx, noise)
            if train:
                return out
            return out.float().numpy()

    def logits_torch(self, idx, noise=0.0):
        """no_grad 前向，返回 torch.Tensor（不转 numpy）——供 mesh 的集成
        统计（ens_acc/ens_nll）在 torch 内融合计算，避免 (B,L,V) 的 numpy
        物化（V=16K 时 ~200MB/批）。精度策略与 forward_torch 一致。"""
        with torch.no_grad():
            amp = bool(getattr(self.cfg, "amp_bf16", False)) if self.cfg is not None else False
            if amp:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    return self._fwd(idx, noise)
            return self._fwd(idx, noise)

    def _fwd(self, idx, noise):
        t = torch.as_tensor(idx, dtype=torch.long)
        B, L = t.shape
        bias = self._causal[:L, :L]
        x = self.tok_emb[t] + self.pos_emb[:L]                       # (B,L,D)
        if noise > 0.0:
            x = x + torch.from_numpy(
                self.rng.normal(0.0, noise, x.shape).astype(np.float32))
        for l in range(self.n_layers):
            h = F.layer_norm(x, (self.D,))
            h2 = h.reshape(B * L, self.D)
            qkv = h2 @ getattr(self, f"l{l}_wqkv") + getattr(self, f"l{l}_bqkv")
            qkv = qkv.reshape(B, L, 3, self.H, self.d_head).permute(2, 0, 3, 1, 4)
            qb = qkv[0].reshape(B * self.H, L, self.d_head)
            kb = qkv[1].reshape(B * self.H, L, self.d_head)
            vb = qkv[2].reshape(B * self.H, L, self.d_head)
            att = F.softmax(
                (qb @ kb.transpose(-1, -2)) / np.sqrt(self.d_head) + bias,
                dim=-1)                                              # (B*H,L,L)
            ctxv = att @ vb                                          # (B*H,L,Dh)
            ctxv = ctxv.reshape(B, self.H, L, self.d_head).permute(0, 2, 1, 3)
            ctxv = ctxv.reshape(B * L, self.D)
            x = x + (ctxv @ getattr(self, f"l{l}_wo")
                     + getattr(self, f"l{l}_bo")).reshape(B, L, self.D)
            h = F.layer_norm(x, (self.D,)).reshape(B * L, self.D)
            ff = F.gelu(h @ getattr(self, f"l{l}_w1")
                        + getattr(self, f"l{l}_b1")) \
                @ getattr(self, f"l{l}_w2") + getattr(self, f"l{l}_b2")
            x = x + ff.reshape(B, L, self.D)
        h = F.layer_norm(x, (self.D,)).reshape(B * L, self.D)
        logits = h @ self.tok_emb.t()                               # 权重共享
        return logits.reshape(B, L, self.V)

    def signature(self, idx):
        """概念签名：冻结投影 sig_emb 的 token 均值池化 -> (B, D) numpy。
        与 numpy 版逐行同构（sig_emb 是共享的冻结 numpy 数组）。"""
        return self.sig_emb[idx].mean(axis=1)

    def forward_stats(self, idx, mask=None, tgt=None):
        """路由统计前向（免物化 + 单趟融合）：mesh 的熵缓存/探针只需三个
        标量统计，不该把 (B,L,V) 的 softmax 搬进 numpy——V=16K 时那是
        400MB/细胞/步（实测 7.5s/step 的元凶）。全部在 torch 里分块算：
        - dist : 平均 next-token 分布 (V,)（分歧路由的锚点熵用）
        - top1 : 输出段 top-1 概率均值（反射弧置信度用）
        - probe: 输出段精度（合同授予探针；tgt 为 None 时不计算）

        融合要点（D=256/V=16384 实测 1123ms -> ~600ms/细胞）：
        1. _fwd 包 autocast（bf16 矩阵乘），与训练前向同精度策略；
        2. logits 保持低精度不整体升 fp32，分块时再 .float()；
        3. 每块一次 softmax（内部融合 max/sub/exp/sum/div）+ 一次
           p.max(-1)（一趟同时取 top1 概率与 argmax）+ 一次 p.sum(0)，
           取代旧版 softmax 分块 / max / logsumexp 全量物化 / argmax
           四趟独立扫描（logsumexp 一项就 437ms/细胞）。
        """
        with torch.no_grad():
            amp = bool(getattr(self.cfg, "amp_bf16", False)) if self.cfg is not None else False
            if amp:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = self._fwd(idx, 0.0)
            else:
                logits = self._fwd(idx, 0.0)
            B, L, V = logits.shape
            n = B * L
            flat = logits.reshape(n, V)
            acc_dist = torch.zeros(V)
            top1_sum, top1_n = 0.0, 0
            correct, count = 0, 0
            row_ok = tgt_flat = None
            if tgt is not None:
                m = np.ones(L, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
                row_ok = np.tile(m, B)
                tgt_flat = np.asarray(tgt).reshape(-1)
            CH = 2048
            for i in range(0, n, CH):
                blk = flat[i:i + CH].float()
                p = blk.softmax(-1)
                acc_dist += p.sum(0)
                v, pred = p.max(-1)          # 一趟同时拿 top1 概率与 argmax
                top1_sum += float(v.sum())
                top1_n += blk.shape[0]
                if tgt_flat is not None:
                    sel = row_ok[i:i + CH]
                    if sel.any():
                        correct += int((pred.numpy()[sel]
                                        == tgt_flat[i:i + CH][sel]).sum())
                        count += int(sel.sum())
            out = dict(dist=(acc_dist / n).numpy(),
                       top1=top1_sum / max(top1_n, 1))
            if tgt_flat is not None:
                out["probe"] = correct / max(count, 1)
            return out

    # ---------- 合同（与 numpy MicroCell 逐行同构） ----------
    def global_reliability(self):
        if not self.seg_rel:
            return 0.5
        return float(np.mean(list(self.seg_rel.values())))

    @property
    def reliability(self):
        return self.global_reliability()

    def cond_reliability(self, ctx):
        # 未记账段先验 0.0（未证明 = 不可靠）。历史上这里是 0.5，理由是
        # "没在异任务上失败过"——但 pilot 实测：从未受训的通才细胞拿 0.5
        # 默认分，与难域专才的实测值（en_wiki 0.527、code 0.056 精度磨蚀
        # 后的低信誉）持平甚至反超，选路坍缩到通才（code_py 路由 ppl
        # 97861 vs 专才 1332）。0.0 先验下，任何有实测证据的细胞都排在
        # 从未受训者之前——专才的"复兴中的前任"（rel 0.21）也不再被
        # 自由细胞（0.5）压制，复现授予的候选排序回归证据驱动。
        if not self.seg_rel:
            return 0.0
        num, wsum = 0.0, 0.0
        for seg_id, w in ctx.items():
            if w <= 0.0:
                continue
            num += w * self.seg_rel.get(seg_id, 0.0)
            wsum += w
        return num / wsum if wsum > 1e-9 else 0.0

    def note(self, seg_id, ok, alpha):
        r = self.seg_rel.get(seg_id, 0.5)
        self.seg_rel[seg_id] = r + alpha * (ok - r)
        self.seg_best[seg_id] = max(self.seg_best.get(seg_id, 0.0), self.seg_rel[seg_id])
        self.seg_note[seg_id] = -1

    def erode(self, seg_id, now, decay, best_decay=None):
        last = self.seg_note.get(seg_id)
        if last is None or last < 0:
            return
        dt = now - last
        if dt > 0:
            self.seg_rel[seg_id] *= decay ** dt
            if best_decay is not None:
                self.seg_best[seg_id] = self.seg_best.get(seg_id, 0.0) * best_decay ** dt
            self.seg_note[seg_id] = now

    def freeze(self, seg_id, end_step, decay, best_decay=None):
        last = self.seg_note.get(seg_id)
        if last is None or last < 0:
            return
        dt = max(0, end_step - last)
        if dt > 0:
            self.seg_rel[seg_id] *= decay ** dt
            if best_decay is not None:
                self.seg_best[seg_id] = self.seg_best.get(seg_id, 0.0) * best_decay ** dt
        self.seg_note[seg_id] = end_step

    def niche(self):
        if not self.seg_rel:
            return 0.5
        return float(max(self.seg_rel.values()))

    # ---------- checkpoint ----------
    def state(self):
        """可序列化的细胞状态（参数 + Adam 矩 + 合同账本；sig_emb 由 mesh 另存）。
        Adam 矩必须入库——否则恢复后首步更新即与保存点轨迹分叉。
        所有数组一律拷贝：.numpy()/from_numpy 都与源存储共享，浅引用会让
        "保存后继续训练"经 Adam 就地更新污染恢复体（实测往返不一致的根因）。"""
        opt = None
        if self._opt is not None:
            opt = dict(state={k: {kk: (vv.clone() if hasattr(vv, "clone") else vv)
                                  for kk, vv in v.items()}
                              for k, v in self._opt.state_dict()["state"].items()},
                       param_groups=[dict(g) for g in
                                     self._opt.state_dict()["param_groups"]])
        return dict(
            params={k: v.detach().cpu().numpy().copy()
                    for k, v in self.named_parameters()},
            opt=opt,
            cell_id=self.cell_id, birth_step=self.birth_step,
            seg_rel=dict(self.seg_rel), seg_note=dict(self.seg_note),
            seg_best=dict(self.seg_best),
        )

    def load_state(self, st):
        for k, v in st["params"].items():
            getattr(self, k).data = torch.from_numpy(
                np.array(v, dtype=np.float32))     # np.array 强制拷贝
        self.seg_rel = dict(st["seg_rel"])
        self.seg_note = dict(st["seg_note"])
        self.seg_best = dict(st["seg_best"])
        self.birth_step = st.get("birth_step", self.birth_step)
        if st.get("opt") is not None:
            # 参数已就地替换（Parameter 对象不变），优化器引用仍有效
            self._optimizer().load_state_dict(st["opt"])
        else:
            self._opt = None
