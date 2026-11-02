"""基线模型：单个 MLP + 全局反向传播 + 在线 SGD（代表"传统大模型"的对照物）。

公平性处理：与 OBM 共用同一套冻结感官层（相同种子初始化），
且隐藏层宽度调到 128，使其可学习参数量（≈4359）与 OBM 满配
（8 细胞 × 547 ≈ 4376）基本相等 —— 比较的是"学习范式"而非参数量。
"""
import numpy as np


class SingleNet:
    def __init__(self, d_in=8, n_hidden=128, n_classes=3, lr=0.05, seed=0):
        self.rng = np.random.default_rng(seed)
        # 与 OBM 相同的冻结感官层（3 个 8×10 随机投影）
        self.sensory = [self.rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, 10))
                        for _ in range(3)]
        pd = 30
        self.W1 = self.rng.normal(0, 1 / np.sqrt(pd), (pd, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = self.rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.lr = lr
        self.energy = 0.0
        self.history = []
        self.step = 0

    def perceive(self, x):
        return np.concatenate([np.tanh(x @ W) for W in self.sensory])

    def _forward(self, p):
        h = np.tanh(p @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum(), h

    def predict(self, p):
        prob, _ = self._forward(p)
        return prob

    def run_step(self, x, y):
        p = self.perceive(x)
        probs, h = self._forward(p)
        pred = int(np.argmax(probs))
        surprise = float(-np.log(probs[y] + 1e-12))
        # 全局反向传播（对照物特征）：所有参数一起更新
        dlogits = probs
        dlogits[y] -= 1.0
        gW2 = np.outer(h, dlogits)
        gb2 = dlogits
        dh = (dlogits @ self.W2.T) * (1 - h * h)
        gW1 = np.outer(p, dh)
        gb1 = dh
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        # 能耗按参数激活量计（全参数激活）
        self.energy += self.W1.size + self.b1.size + self.W2.size + self.b2.size
        rec = dict(step=self.step, correct=int(pred == y), surprise=surprise,
                   k=1, reflex=0, n_cells=1,
                   n_params=self.W1.size + self.b1.size + self.W2.size + self.b2.size,
                   energy=self.energy)
        self.history.append(rec)
        self.step += 1
        return rec

    def evaluate(self, test_sets):
        accs = []
        for X, y in test_sets:
            ok = sum(int(np.argmax(self.predict(self.perceive(x))) == yy)
                     for x, yy in zip(X, y))
            accs.append(ok / len(y))
        return accs
