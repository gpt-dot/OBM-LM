"""fast_cell.py —— OBM-LM 的「快训」架构：共享 trunk + 薄专才 head。

设计目标（用户要求：同性质、低算力、快进度、快 tok'）：
  原架构 = N 个完全独立的全量 transformer 细胞。每步训练 = 1 个全量前向+
  反向（受授细胞），每 N 步还要全库 N× 前向刷新熵缓存；推理 = N× 全量前向
  （全库锚点）+ k× 全量前向（集成）。规模一大，N× 与「全量」同时爆炸。

  本架构把"规模"与"专才"解耦：
    * 共享 trunk：一个 LLM 量级的大 transformer（规模在此），所有细胞共用，
      前向只算一次；训练期冻结（或极小 lr），只提供通用表征。
    * 薄专才 head：每个细胞 = 1~2 个 transformer block + LM 头（权重共享到
      trunk.tok_emb），参数量约为全量的 5~10%。OBM 的全部机制（概念签名 /
      条件合同市场 / 双信号边界 / 单份+复现授予 / 梦境代谢 / 记账存疑）都
      跑在 head 上——细胞"学什么、专不专"由 head 决定，trunk 只提供容量。

  速度来源（与 TRIZ 矛盾分解对应）：
    1) 训练：每步只做 1 次 trunk 前向（冻结、无 grad）+ 1 次薄 head 前向/反向
       （受授细胞），不再有"N× 全量前向"；熵缓存刷新也可省（路由靠签名+seg_rel
       表，不需全细胞实时前向）——较原架构 ~N× 加速。
    2) 推理：1 次 trunk 前向 + k(≤3) 次薄 head 前向，替代原 N× 全量前向——
       tok' 输出 ~N× 加速。
    3) 规模不变：trunk 可做到 d=768/L=10（LLM 量级），总参数仍为 LLM 量级。

  与 OBM 机制完全兼容：FastCell 继承 TorchMicroLM 的全部接口（signature /
  cond_reliability / note / erode / freeze / spawn / state / load_state /
  forward_stats / adam_step ...），mesh（OBMLM）零改动复用——只是细胞"身体"
  从全量 transformer 换成 trunk+head。trunk 不注册为 cell 的子模块（用
  list 容器持有），故 n_params/parameters/optimizer 只包含 head 参数。
"""
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_cell import TorchMicroLM


class SharedTrunk(nn.Module):
    """共享主干 transformer（规模所在）。冻结前向默认 no_grad；pretrain 期
    用 train=True 开 grad 单独训练，之后冻结给 OBM 用。"""

    def __init__(self, rng, vocab, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.V, self.D, self.H = vocab, d_model, n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers
        self.L_max = max_len
        s = 1.0 / math.sqrt(d_model)
        # 概念签名投影：冻结 numpy（与细胞共享同一份，领地几何一致）
        self.sig_emb = rng.normal(0.0, 1.0, (vocab, d_model)).astype(np.float32)
        self.tok_emb = nn.Parameter(
            torch.from_numpy(rng.normal(0.0, 1.0, (vocab, d_model)).astype(np.float32)))
        self.pos_emb = nn.Parameter(
            torch.from_numpy(rng.normal(0.0, 0.02, (max_len, d_model)).astype(np.float32)))
        for l in range(n_layers):
            setattr(self, f"l{l}_wqkv",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (d_model, 3 * d_model)).astype(np.float32))))
            setattr(self, f"l{l}_bqkv", nn.Parameter(torch.zeros(3 * d_model)))
            setattr(self, f"l{l}_wo",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (d_model, d_model)).astype(np.float32))))
            setattr(self, f"l{l}_bo", nn.Parameter(torch.zeros(d_model)))
            setattr(self, f"l{l}_w1",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (d_model, 2 * d_model)).astype(np.float32))))
            setattr(self, f"l{l}_b1", nn.Parameter(torch.zeros(2 * d_model)))
            setattr(self, f"l{l}_w2",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (2 * d_model, d_model)).astype(np.float32))))
            setattr(self, f"l{l}_b2", nn.Parameter(torch.zeros(d_model)))
        self._causal = torch.triu(
            torch.full((max_len, max_len), -1e9), diagonal=1)
        self.Ln = nn.LayerNorm(d_model)

    def forward(self, idx, train=False):
        t = torch.as_tensor(idx, dtype=torch.long)
        B, L = t.shape
        bias = self._causal[:L, :L]
        x = self.tok_emb[t] + self.pos_emb[:L]
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for l in range(self.n_layers):
                h = F.layer_norm(x, (self.D,))
                qkv = h.reshape(B * L, self.D) @ getattr(self, f"l{l}_wqkv") \
                    + getattr(self, f"l{l}_bqkv")
                qkv = qkv.reshape(B, L, 3, self.H, self.d_head).permute(2, 0, 3, 1, 4)
                qb = qkv[0].reshape(B * self.H, L, self.d_head)
                kb = qkv[1].reshape(B * self.H, L, self.d_head)
                vb = qkv[2].reshape(B * self.H, L, self.d_head)
                att = F.softmax((qb @ kb.transpose(-1, -2)) / math.sqrt(self.d_head)
                               + bias, dim=-1)
                cv = att @ vb
                cv = cv.reshape(B, self.H, L, self.d_head).permute(0, 2, 1, 3) \
                    .reshape(B * L, self.D)
                x = x + (cv @ getattr(self, f"l{l}_wo")
                         + getattr(self, f"l{l}_bo")).reshape(B, L, self.D)
                h = F.layer_norm(x, (self.D,)).reshape(B * L, self.D)
                ff = F.gelu(h @ getattr(self, f"l{l}_w1")
                            + getattr(self, f"l{l}_b1")) \
                    @ getattr(self, f"l{l}_w2") + getattr(self, f"l{l}_b2")
                x = x + ff.reshape(B, L, self.D)
            h = self.Ln(x)
        return h  # (B, L, D) — 冻结期无 grad

    def n_params(self):
        return sum(q.numel() for q in self.parameters())


class FastCell(TorchMicroLM):
    """薄专才细胞：共享 trunk + 自有薄 head（L_head 个 block + LM 头）。

    继承 TorchMicroLM 全部接口；只重写"前向/统计/参数/优化/分裂"，使它们走
    trunk+head 而非全量 transformer。trunk 经 list 容器持有（不注册为子模块），
    故 n_params/parameters/optimizer 仅含 head 参数——mesh 对细胞的账本、授予、
    梦境逻辑完全不变。
    """

    backend = "torch"

    def __init__(self, rng, vocab, trunk, d_model, n_heads, n_layers_head,
                 max_len, cell_id=0, cfg=None):
        nn.Module.__init__(self)  # 不调 TorchMicroLM.__init__（那会建全量 transformer）
        self.cell_id = cell_id
        self.rng = rng
        self.V, self.D, self.H, self.L_max = vocab, d_model, n_heads, max_len
        self.n_layers = n_layers_head
        self.d_head = d_model // n_heads
        self.cfg = cfg
        self.sig_emb = trunk.sig_emb          # 冻结签名投影（与 trunk 共享）
        self._trunk_holder = [trunk]          # list 容器：避免注册为子模块
        s = 1.0 / math.sqrt(d_model)
        # 薄 head：n_layers_head 个 block（结构同 trunk block）
        for l in range(n_layers_head):
            setattr(self, f"h{l}_wqkv",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (d_model, 3 * d_model)).astype(np.float32))))
            setattr(self, f"h{l}_bqkv", nn.Parameter(torch.zeros(3 * d_model)))
            setattr(self, f"h{l}_wo",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (d_model, d_model)).astype(np.float32))))
            setattr(self, f"h{l}_bo", nn.Parameter(torch.zeros(d_model)))
            setattr(self, f"h{l}_w1",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (d_model, 2 * d_model)).astype(np.float32))))
            setattr(self, f"h{l}_b1", nn.Parameter(torch.zeros(2 * d_model)))
            setattr(self, f"h{l}_w2",
                    nn.Parameter(torch.from_numpy(
                        rng.normal(0.0, s, (2 * d_model, d_model)).astype(np.float32))))
            setattr(self, f"h{l}_b2", nn.Parameter(torch.zeros(d_model)))
        self.birth_step = 0
        self.seg_rel = {}
        self.seg_note = {}
        self.seg_best = {}
        self.activations = 0
        self.correct = 0
        self._opt = None
        self._causal = trunk._causal
        self._trunk_train = False      # 由 OBMLM_Fast 在开启 trunk-lr 时置 True

    # trunk 经 property 暴露（list 容器持有，不污染 parameters()）
    @property
    def trunk(self):
        return self._trunk_holder[0]

    def _trunk_h(self, idx, train=False):
        # 默认冻结（无 grad）。仅当开启 trunk-lr 且处于训练路径时，才让梯度
        # 流经共享 trunk——这样 trunk 随 OBM 进化（每步只算 1 次，速度优势保住），
        # 而 eval/统计路径恒 no_grad，不污染。
        if self._trunk_train and train:
            return self.trunk(idx, train=True)
        with torch.no_grad():
            return self.trunk(idx)            # 冻结，无 grad

    def _head(self, h, train):
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            x = h
            B, L, _ = x.shape
            bias = self._causal[:L, :L]
            for l in range(self.n_layers):
                hh = F.layer_norm(x, (self.D,))
                qkv = hh.reshape(B * L, self.D) @ getattr(self, f"h{l}_wqkv") \
                    + getattr(self, f"h{l}_bqkv")
                qkv = qkv.reshape(B, L, 3, self.H, self.d_head).permute(2, 0, 3, 1, 4)
                qb = qkv[0].reshape(B * self.H, L, self.d_head)
                kb = qkv[1].reshape(B * self.H, L, self.d_head)
                vb = qkv[2].reshape(B * self.H, L, self.d_head)
                att = F.softmax((qb @ kb.transpose(-1, -2)) / math.sqrt(self.d_head)
                               + bias, dim=-1)
                cv = att @ vb
                cv = cv.reshape(B, self.H, L, self.d_head).permute(0, 2, 1, 3) \
                    .reshape(B * L, self.D)
                x = x + (cv @ getattr(self, f"h{l}_wo")
                         + getattr(self, f"h{l}_bo")).reshape(B, L, self.D)
                hh = F.layer_norm(x, (self.D,)).reshape(B * L, self.D)
                ff = F.gelu(hh @ getattr(self, f"h{l}_w1")
                            + getattr(self, f"h{l}_b1")) \
                    @ getattr(self, f"h{l}_w2") + getattr(self, f"h{l}_b2")
                x = x + ff.reshape(B, L, self.D)
            hh = F.layer_norm(x, (self.D,)).reshape(B * L, self.D)
            logits = hh @ self.trunk.tok_emb.t()   # 权重共享到 trunk 词嵌入
        return logits.reshape(B, L, self.V)

    # ---------- 前向（覆盖基类：trunk 默认 no_grad，仅 trunk_train+train 时走 grad）----------
    def forward_torch(self, idx, noise=0.0, train=False):
        h = self._trunk_h(idx, train)
        return self._head(h, train)

    def logits_torch(self, idx, noise=0.0):
        h = self._trunk_h(idx, False)
        return self._head(h, False)

    def _fwd(self, idx, noise):
        h = self._trunk_h(idx, False)
        return self._head(h, False)

    def forward_stats(self, idx, mask=None, tgt=None):
        h = self._trunk_h(idx, False)
        logits = self._head(h, False)
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
            v, pred = p.max(-1)
            top1_sum += float(v.sum())
            top1_n += blk.shape[0]
            if tgt_flat is not None:
                sel = row_ok[i:i + CH]
                if sel.any():
                    correct += int((pred.numpy()[sel]
                                    == tgt_flat[i:i + CH][sel]).sum())
                    count += int(sel.sum())
        out = dict(dist=(acc_dist / n).numpy(), top1=top1_sum / max(top1_n, 1))
        if tgt_flat is not None:
            out["probe"] = correct / max(count, 1)
        return out

    # ---------- 参数 / 优化：仅 head ----------
    def n_params(self):
        return sum(q.numel() for q in self.parameters())

    def param_list(self):
        return list(self.parameters())

    def adam_step(self, lr, clip=0.0):
        opt = self._optimizer()
        for g in opt.param_groups:
            g["lr"] = float(lr)
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    # ---------- 分裂：生成 FastCell 并共享 trunk ----------
    def spawn(self, rng, noise, new_id, birth_step):
        child = FastCell.__new__(FastCell)
        nn.Module.__init__(child)
        child.cell_id = new_id
        child.rng = self.rng
        child.V, child.D, child.H = self.V, self.D, self.H
        child.n_layers, child.d_head, child.L_max = self.n_layers, self.d_head, self.L_max
        child.cfg = self.cfg
        child.sig_emb = self.sig_emb
        child._trunk_holder = [self.trunk]
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
        child._trunk_train = self._trunk_train
        return child

    # state()/load_state() 继承基类：named_parameters() 仅 head，自动正确


from .lm_mesh import OBMLM
import obm_lm.backend as bk


class OBMLM_Fast(OBMLM):
    """共享 trunk 版 mesh：复用 OBMLM 全部机制，仅把细胞换成 FastCell。"""

    def __init__(self, cfg, trunk_layers=None, head_layers=None, trunk_train=False):
        super().__init__(cfg)
        self.trunk_layers = trunk_layers or cfg.n_layers
        self.head_layers = head_layers or 1
        self.trunk_train = trunk_train
        rng = self.rng
        self.trunk = SharedTrunk(rng, cfg.vocab, cfg.d_model, cfg.n_heads,
                                 self.trunk_layers, cfg.max_len)
        self.cells = [FastCell(rng, cfg.vocab, self.trunk, cfg.d_model,
                               cfg.n_heads, self.head_layers, cfg.max_len, i,
                               cfg=cfg) for i in range(cfg.n_cells_init)]
        for c in self.cells:
            c._trunk_train = trunk_train
        self.next_id = cfg.n_cells_init

    def set_trunk_train(self, on):
        """运行时切换 trunk 是否随梯度进化（仅训练路径）。"""
        self.trunk_train = on
        for c in self.cells:
            c._trunk_train = on
        for p in self.trunk.parameters():
            p.requires_grad_(on)

    def checkpoint(self):
        ck = super().checkpoint()   # 细胞 state 已是 head-only
        ck["trunk"] = {name: p.detach().cpu().numpy().copy()
                       for name, p in self.trunk.named_parameters()}
        return ck

    def restore(self, ck):
        super().restore(ck)         # 载入 mesh 级状态（含被丢弃的 TorchMicroLM 细胞）
        # 重建 trunk + FastCell（按 checkpoint 的 head 状态载入）
        self.trunk = SharedTrunk(self.rng, self.cfg.vocab, self.cfg.d_model,
                                 self.cfg.n_heads, self.trunk_layers,
                                 self.cfg.max_len)
        tp = ck["trunk"]
        for name, p in self.trunk.named_parameters():
            p.data = torch.from_numpy(np.array(tp[name], dtype=np.float32))
        self.cells = []
        for st in ck["cells"]:
            c = FastCell(self.rng, self.cfg.vocab, self.trunk, self.cfg.d_model,
                         self.cfg.n_heads, self.head_layers, self.cfg.max_len,
                         st["cell_id"], cfg=self.cfg)
            bk.cell_load(c, st)
            self.cells.append(c)
        for c in self.cells:
            c.rng = self.rng
