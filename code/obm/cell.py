"""MicroCell —— OBM 的最小计算单元（概念细胞）。

每个细胞是一个独立的微型 MLP：
  * 不与其他细胞共享参数；
  * 不交换梯度（没有全局反向传播，只有"细胞内局部梯度"）；
  * 只通过"合同"与外界互动。

MVP v3 关键升级：条件合同 + 领地惰性衰减。
  * 每个细胞按概念段维护一份可靠性（seg_rel）；
  * 推理时按输入与各段统计的匹配度加权取用：
    **只在自己有履约记录的领地上投票，在任何地方都保持被激活学习的资格**；
  * 合同值的时间语义由 mesh 维护：当前段的值随调用实时惰性衰减，
    段封存后即冻结（"不用则衰"只发生在领地活跃而细胞久不履约时）。

这对应设计文档 PC3"基于条件的分离"的代码化。
"""
import numpy as np


class MicroCell:
    __slots__ = ("W1", "b1", "W2", "b2", "cell_id",
                 "d_in", "n_hidden", "n_classes",
                 "activations", "correct", "seg_rel", "seg_note",
                 "birth_step")

    def __init__(self, rng, d_in, n_hidden, n_classes, cell_id=0,
                 W1=None, b1=None, W2=None, b2=None):
        self.d_in, self.n_hidden, self.n_classes = d_in, n_hidden, n_classes
        self.cell_id = cell_id
        if W1 is None:  # 同构模板 + 随机初始化
            self.W1 = rng.normal(0.0, 0.5 / np.sqrt(d_in), (d_in, n_hidden))
            self.b1 = np.zeros(n_hidden)
            self.W2 = rng.normal(0.0, 0.5 / np.sqrt(n_hidden), (n_hidden, n_classes))
            self.b2 = np.zeros(n_classes)
        else:           # 分裂诞生：复制父代参数（外层已做变异）
            self.W1, self.b1 = W1.copy(), b1.copy()
            self.W2, self.b2 = W2.copy(), b2.copy()
        self.activations = 0
        self.correct = 0
        self.seg_rel = {}                # 条件合同：segment_id -> 领地可靠性
        self.seg_note = {}               # segment_id -> 最后一次记账的步数
        self.birth_step = 0

    # ---------- 合同可靠性 ----------
    def global_reliability(self):
        """全局可靠性 = 各领地可靠性的均值（无记录时为中性 0.5）。"""
        if not self.seg_rel:
            return 0.5
        return float(np.mean(list(self.seg_rel.values())))

    @property
    def reliability(self):
        return self.global_reliability()

    def cond_reliability(self, ctx):
        """条件合同可靠性：按输入与各概念段的匹配度加权。

        ctx: {segment_id: weight}（权重和≈1）。对没有履约记录的段，
        回退到该细胞的全局均值——"没去过的地方不表态"。
        """
        if not self.seg_rel:
            return 0.5
        g = self.global_reliability()
        num, wsum = 0.0, 0.0
        for seg_id, w in ctx.items():
            if w <= 0.0:
                continue
            num += w * self.seg_rel.get(seg_id, g)
            wsum += w
        return num / wsum if wsum > 1e-9 else g

    def note(self, seg_id, ok, alpha):
        """在某个概念段的领地上记一次功过（合同记账）。

        调用方（mesh）负责在记账前完成惰性衰减的 catch-up。
        """
        r = self.seg_rel.get(seg_id, 0.5)
        self.seg_rel[seg_id] = r + alpha * (ok - r)
        self.seg_note[seg_id] = -1        # 由 mesh 在 note 后置为当前步

    def erode(self, seg_id, now, decay):
        """领地惰性衰减：把某段的合同值推进到 now（按距上次记账的时长）。

        只对"活跃领地"调用——封存的段由 mesh 在封存时一次性冻结。
        """
        last = self.seg_note.get(seg_id)
        if last is None or last < 0:
            return
        dt = now - last
        if dt > 0:
            self.seg_rel[seg_id] *= decay ** dt
            self.seg_note[seg_id] = now

    def freeze(self, seg_id, end_step, decay):
        """段封存：把合同值冻结在封存时刻（此后再不随时间衰减）。"""
        last = self.seg_note.get(seg_id)
        if last is None or last < 0:
            return
        dt = max(0, end_step - last)
        if dt > 0:
            self.seg_rel[seg_id] *= decay ** dt
        self.seg_note[seg_id] = end_step

    def niche(self):
        """最佳领地可靠性——细胞"安身立命"的那个任务上的表现。"""
        if not self.seg_rel:
            return 0.5
        return float(max(self.seg_rel.values()))

    # ---------- 前向 ----------
    def _hidden(self, p):
        return np.tanh(p @ self.W1 + self.b1)

    def predict(self, p):
        """返回 K 维概率分布（细胞的"观点"）。"""
        h = self._hidden(p)
        logits = h @ self.W2 + self.b2
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

    # ---------- 局部学习 ----------
    def learn(self, p, y, lr):
        """细胞内局部梯度更新（softmax 交叉熵）。

        注意：梯度只在本细胞参数内流动，绝不跨细胞回传 ——
        这是对"全局反向传播"这一最贵部件的 TRIZ 裁剪。
        返回本次损失（用于意外度与蒸馏判断）。
        """
        h = self._hidden(p)
        logits = h @ self.W2 + self.b2
        z = logits - logits.max()
        e = np.exp(z)
        probs = e / e.sum()
        loss = -np.log(probs[y] + 1e-12)

        dlogits = probs
        dlogits[y] -= 1.0
        gW2 = np.outer(h, dlogits)
        gb2 = dlogits
        dh = (dlogits @ self.W2.T) * (1.0 - h * h)
        gW1 = np.outer(p, dh)
        gb1 = dh

        self.W2 -= lr * gW2
        self.b2 -= lr * gb2
        self.W1 -= lr * gW1
        self.b1 -= lr * gb1
        return float(loss)

    # ---------- 分裂（TRIZ 原理 26：复制） ----------
    def spawn(self, rng, noise, new_id, birth_step):
        child = MicroCell(
            rng, self.d_in, self.n_hidden, self.n_classes, new_id,
            W1=self.W1 * (1.0 + noise * rng.normal(size=self.W1.shape)),
            b1=self.b1,
            W2=self.W2 * (1.0 + noise * rng.normal(size=self.W2.shape)),
            b2=self.b2,
        )
        child.birth_step = birth_step
        # 子代继承父代的领地合同，但向中性值阻尼（见习期：
        # 它是突变拷贝，尚未证明自己，不应带着父代的全额信誉投票）
        child.seg_rel = {s: 0.5 + 0.4 * (v - 0.5)
                         for s, v in self.seg_rel.items()}
        child.seg_note = {s: birth_step for s in self.seg_rel}
        return child

    @property
    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def __repr__(self):
        return (f"<Cell#{self.cell_id} g={self.global_reliability():.2f} "
                f"niche={self.niche():.2f} act={self.activations} "
                f"ok={self.correct} segs={len(self.seg_rel)}>")
