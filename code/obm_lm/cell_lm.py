"""MicroLM —— OBM 的语言模型细胞：小型因果 Transformer（纯 numpy + mini-autograd）。

结构（pre-norm，GPT 式权重共享）：
    tok_emb + pos_emb → 2×[LN → 4 头因果自注意力 → 残差 ; LN → FFN → 残差]
    → LN → logits = h @ tok_emb.T

~69K 参数/细胞（D=64, H=4, L=24, V=16）。所有矩阵乘走 2D/bmm，
广播由 autograd 的 add/mul 处理；因果 mask 是加在注意力分数上的常量偏置。

概念签名用**独立的冻结投影** sig_emb（不训练，mesh 级共享）：token 均值池化。
MVP 的教训——领地识别必须匹配"输入是什么"而不是"模型现在是什么状态"，
训练中的 embedding 漂移会让冻结参照失效（实测段内距离从 2.9 漂到 32）。
合同（seg_rel / cond_reliability / note / erode / freeze / niche）与 MVP 的
MicroCell 逐行同构——专才化机制在语言规模原样复用。
"""
import numpy as np
from obm_lm.autograd import T, bmm, gather, softmax, layernorm, gelu


class MicroLM:
    """一个语言模型细胞。forward 返回 logits（autograd 图，训练时 backward）。"""

    def __init__(self, rng, vocab, d_model=64, n_heads=4, n_layers=2,
                 max_len=32, cell_id=0, sig_emb=None):
        self.cell_id = cell_id
        self.rng = rng                # 反事实噪声用
        self.V, self.D, self.H, self.L_max = vocab, d_model, n_heads, max_len
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        # 概念签名投影：冻结（不训练），mesh 级共享；单独使用时自建
        self.sig_emb = sig_emb if sig_emb is not None else \
            rng.normal(0.0, 1.0, (vocab, d_model))
        s = 1.0 / np.sqrt(d_model)
        P = {}
        P["tok_emb"] = T(rng.normal(0, 1.0, (vocab, d_model)))
        P["pos_emb"] = T(rng.normal(0, 0.02, (max_len, d_model)))
        for l in range(n_layers):
            p = f"l{l}."
            P[p + "wqkv"] = T(rng.normal(0, s, (d_model, 3 * d_model)))
            P[p + "bqkv"] = T(np.zeros(3 * d_model))
            P[p + "wo"] = T(rng.normal(0, s, (d_model, d_model)))
            P[p + "bo"] = T(np.zeros(d_model))
            P[p + "w1"] = T(rng.normal(0, s, (d_model, 2 * d_model)))
            P[p + "b1"] = T(np.zeros(2 * d_model))
            P[p + "w2"] = T(rng.normal(0, s, (2 * d_model, d_model)))
            P[p + "b2"] = T(np.zeros(d_model))
        self.params = P
        self.birth_step = 0
        # 条件合同（与 MVP 的 MicroCell 同构）
        self.seg_rel = {}      # segment_id -> 领地可靠性
        self.seg_note = {}     # segment_id -> 最后一次记账的步
        # 承诺度峰值：seg_best[sid] = 该段曾有过的最高信誉（慢衰减）。
        # 用途：合同授予时判定"专才"（有存量知识要保护）用峰值而非当前值——
        # 切换窗口内 3 个违约记账就能把 r 从 0.9 砍到 0.66，若按当前值判定
        # 专才会被误判为"自由细胞"而被授予异乡合同（实测 copy 专才就这样
        # 被转化成 reverse 专才）。峰值衰减（0.999/步）远慢于可靠性衰减
        # （0.9995/步）：扛得住 ~50 步的记账污染窗口，又能在 ~250 步异段
        # 缺席后释放细胞，保证任务复现时旧专家能被重新授予合同。
        self.seg_best = {}
        self.activations = 0
        self.correct = 0
        self._causal_bias = np.triu(np.full((max_len, max_len), -1e9), k=1)

    # ---------- 参数 ----------
    def param_list(self):
        return list(self.params.values())

    def n_params(self):
        return sum(p.data.size for p in self.params.values())

    def spawn(self, rng, noise, new_id, birth_step):
        """分裂：复制父代参数 + 噪声变异（MVP 同款）。"""
        child = MicroLM.__new__(MicroLM)
        child.cell_id = new_id
        child.rng = self.rng
        child.V, child.D, child.H = self.V, self.D, self.H
        child.n_layers, child.d_head, child.L_max = self.n_layers, self.d_head, self.L_max
        child.params = {k: T(v.data + rng.normal(0, noise, v.data.shape))
                        for k, v in self.params.items()}
        child.sig_emb = self.sig_emb      # 冻结投影：共享同一份（不复制）
        child.birth_step = birth_step
        child.seg_rel = {}
        child.seg_note = {}
        child.seg_best = {}
        child.activations = 0
        child.correct = 0
        child._causal_bias = self._causal_bias
        return child

    # ---------- checkpoint ----------
    def state(self):
        """可序列化的细胞状态（参数 + Adam 矩 + 合同账本）。"""
        opt = getattr(self, "_opt", {})
        return dict(
            params={k: p.data.copy() for k, p in self.params.items()},
            opt={k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in opt.items()},
            cell_id=self.cell_id, birth_step=self.birth_step,
            seg_rel=dict(self.seg_rel), seg_note=dict(self.seg_note),
            seg_best=dict(self.seg_best))

    def load_state(self, st):
        for k, v in st["params"].items():
            self.params[k].data[...] = v
        self._opt = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in st.get("opt", {}).items()}
        self.seg_rel = dict(st["seg_rel"])
        self.seg_note = dict(st["seg_note"])
        self.seg_best = dict(st["seg_best"])
        self.birth_step = st.get("birth_step", self.birth_step)

    # ---------- 前向 ----------
    @staticmethod
    def _pick(t, i):
        """取第 0 维第 i 片并压掉该维（纯切片，autograd 安全）。"""
        return t[i:i + 1].reshape(t.shape[1:])

    def forward(self, idx, noise=0.0):
        """idx: (B, L) int -> logits (B, L, V)（autograd T）。

        noise > 0 时对输入 embedding 加高斯噪（梦境反事实考题）。
        """
        B, L = idx.shape
        P = self.params
        bias = self._causal_bias[:L, :L]
        x = gather(P["tok_emb"], idx) + P["pos_emb"][:L]        # (B,L,D)
        if noise > 0.0:
            x = x + T(self.rng.normal(0.0, noise, x.shape))
        for l in range(self.n_layers):
            p = f"l{l}."
            h = layernorm(x)
            h2 = h.reshape(B * L, self.D)
            qkv = (h2 @ P[p + "wqkv"] + P[p + "bqkv"])          # (B*L, 3D)
            qkv = qkv.reshape(B, L, 3, self.H, self.d_head).transpose(2, 0, 1, 3, 4)
            qb = self._pick(qkv, 0).transpose(0, 2, 1, 3).reshape(B * self.H, L, self.d_head)
            kb = self._pick(qkv, 1).transpose(0, 2, 1, 3).reshape(B * self.H, L, self.d_head)
            vb = self._pick(qkv, 2).transpose(0, 2, 1, 3).reshape(B * self.H, L, self.d_head)
            att = softmax(bmm(qb, kb.transpose(0, 2, 1)) / np.sqrt(self.d_head) + bias,
                          axis=-1)                              # (B*H,L,L)
            ctx = bmm(att, vb)                                  # (B*H,L,Dh)
            ctx = ctx.reshape(B, self.H, L, self.d_head).transpose(0, 2, 1, 3)
            ctx = ctx.reshape(B * L, self.D)
            x = x + (ctx @ P[p + "wo"] + P[p + "bo"]).reshape(B, L, self.D)
            h = layernorm(x).reshape(B * L, self.D)
            ff = gelu(h @ P[p + "w1"] + P[p + "b1"]) @ P[p + "w2"] + P[p + "b2"]
            x = x + ff.reshape(B, L, self.D)
        h = layernorm(x).reshape(B * L, self.D)
        logits = h @ P["tok_emb"].transpose()                   # (B*L, V) 权重共享
        return logits.reshape(B, L, self.V)
    def signature(self, idx):
        """概念签名：冻结投影 sig_emb 的 token 均值池化 -> (B, D)。

        领地匹配在这个空间做——不同任务的 token 统计不同，签名分布即可区分
        （MVP"输入空间分布匹配"在 LM 的类比）。sig_emb 冻结不训练：
        训练中的 tok_emb 漂移会让段内距离失控（实测 2.9 -> 32），
        而领地识别要回答的是"输入是什么"而非"模型现在是什么状态"。
        """
        return self.sig_emb[idx].mean(axis=1)                     # (B, D)

    # ---------- 合同（与 MVP MicroCell 逐行同构） ----------
    def global_reliability(self):
        if not self.seg_rel:
            return 0.5
        return float(np.mean(list(self.seg_rel.values())))

    @property
    def reliability(self):
        return self.global_reliability()

    def cond_reliability(self, ctx):
        """条件可靠性 = ctx 加权的段可靠性。

        未记账段先验 0.0——"未证明 = 不可靠"。演进：MVP 的全局均值
        泄漏（专才凭 ≈0.7 的全局均值混进异任务池被覆写）→ 先验 0.5
        堵住均值泄漏，但 pilot 实测 0.5 仍高于难域专才磨蚀后的实测值，
        从未受训的通才细胞凭默认分与专才持平反超，选路坍缩（code_py
        路由 ppl 97861 vs 专才 1332）。0.0 先验下所有排序回归实测
        证据驱动，同时保留堵漏效果（0.0 远低于合同门限 0.55）。
        """
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
        self.seg_note[seg_id] = -1        # 由 mesh 在 note 后置为当前步

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
