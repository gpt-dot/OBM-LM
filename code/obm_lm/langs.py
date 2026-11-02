"""langs.py —— OBM-LM 的概念漂移数据流：合成形式语言。

vocab=16：0=BOS，15=SEP，1-14 数据 token。
四个任务用**互不相交的 token 范围**（模拟真实场景中不同领域的词汇差异，
同时保证概念签名 = token 频率统计即可区分——MVP 的教训是签名必须可分）：

  copy     s∈1..14 -> 原样复制        （需要注意力：偏移复制）
  reverse  s∈1..7  -> 逆序            （需要反向偏移）
  sort     s∈8..14 -> 升序排列        （需要比较+重排，最难）
  shift    s∈1..9  -> 逐 token +1     （纯映射，最简单，无需注意力）

PLAN 顺序到达，后两段是 copy/reverse **复现**（同规格同 seed 采样），
用于检验任务复现时的段级合并与合同复兴。

序列格式：[BOS] inp [SEP] out，长度 2*6+2=14。
LM 目标 = 右移一位；**输出段指标**只算 SEP 之后的位置（预测 out），
输入段的复制是平凡任务不能算分。
"""
import numpy as np

BOS, SEP = 0, 15
V = 16
S_LEN = 6
SEQ_LEN = 2 * S_LEN + 2

TASKS = ["copy", "reverse", "sort", "shift"]
PLAN = ["copy", "reverse", "sort", "shift", "copy", "reverse"]   # 后两个为复现


def _gen(task, rng, n):
    """生成 n 条 (序列, 目标, 输出段位置 mask)。"""
    if task == "copy":
        s = rng.integers(1, 15, size=(n, S_LEN))
        out = s.copy()
    elif task == "reverse":
        s = rng.integers(1, 8, size=(n, S_LEN))
        out = s[:, ::-1].copy()
    elif task == "sort":
        s = rng.integers(8, 15, size=(n, S_LEN))
        out = np.sort(s, axis=1)
    elif task == "shift":
        s = rng.integers(1, 10, size=(n, S_LEN))
        out = np.where(s >= 9, 1, s + 1)
    else:
        raise ValueError(task)
    seq = np.concatenate([np.full((n, 1), BOS), s, np.full((n, 1), SEP), out], axis=1)
    tgt = np.concatenate([s, np.full((n, 1), SEP), out,
                          np.zeros((n, 1), dtype=int)], axis=1)
    # 输出段 mask：SEP 之后的位置（预测 out 的每个 token）
    mask = np.zeros(SEQ_LEN, dtype=bool)
    mask[S_LEN + 2:] = True                     # 位置 8..13 预测 out[0..5]
    return seq.astype(int), tgt.astype(int), mask


class LangBlock:
    """一个概念段：规格 + rng（训练/测试/复现共享规格，采样独立）。"""

    def __init__(self, name, seed):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def batch(self, bs):
        return _gen(self.name, self.rng, bs)

    def testset(self, n):
        rng = np.random.default_rng(9999)        # 测试集固定 seed，可复现
        return _gen(self.name, rng, n)


def build_blocks(n_train, n_test):
    """返回 (blocks, test_sets, names)。

    blocks: [LangBlock]*6（顺序到达）；test_sets: 每个**去重任务**的测试集。
    """
    blocks = [LangBlock(name, 1000 + i) for i, name in enumerate(PLAN)]
    seen, test_sets = {}, []
    for b in blocks:
        if b.name not in seen:
            seen[b.name] = len(test_sets)
            test_sets.append(b.testset(n_test))
    names = [b.name for b in blocks if b.name in
             {n for i, n in enumerate(PLAN) if i == PLAN.index(n)}]
    # 上面 names 构造保持 PLAN 首次出现顺序
    names = []
    for n in PLAN:
        if n not in names:
            names.append(n)
    return blocks, test_sets, names


def output_acc(logits_probs, tgt, mask):
    """输出段 next-token 准确率：logits_probs (B,L,V) 已 softmax 或直接 argmax。"""
    pred = np.argmax(logits_probs, axis=-1)
    return float(np.mean(pred[:, mask] == tgt[:, mask]))
