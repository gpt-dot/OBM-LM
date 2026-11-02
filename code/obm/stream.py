"""概念漂移数据流 —— 持续学习的最小试金石。

7 个概念段顺序到达，其中第 5、6 段是第 1、2 段的"复现"：
  1. linear  线性可分二分类（任务规格：固定超平面）
  2. xor     异或（2 个信息维）
  3. wave    波浪边界（非线性）
  4. blobs3  三分类高斯团（任务规格：固定团心）
  5. linear  复现（同一超平面，检验旧任务保持）
  6. xor     复现
  7. moons   双月牙（新概念）

每个概念段：400 个训练样本（在线单次流过）+ 200 个固定测试样本。
关键：同一概念的"任务规格"（超平面/团心）在所有块间共享 ——
训练集、测试集、复现段是同一个任务的不同样本，否则测的是噪声。
系统不知道段边界（OBM 靠自己检测），但评估脚本知道（用于计算保持率）。
"""
import numpy as np

D_IN = 8


# ---------- 任务规格（同一概念的所有块共享） ----------
def _linear_spec(rng):
    return dict(w=rng.normal(size=D_IN), b=rng.normal() * 0.5)


def _xor_spec(rng):
    return dict()


def _wave_spec(rng):
    return dict()


def _blobs3_spec(rng):
    return dict(centers=rng.normal(scale=3.0, size=(3, D_IN)))


def _moons_spec(rng):
    return dict()


SPECS = {"linear": _linear_spec, "xor": _xor_spec, "wave": _wave_spec,
         "blobs3": _blobs3_spec, "moons": _moons_spec}


# ---------- 采样器 ----------
def _linear(spec, rng, n):
    X = rng.normal(size=(n, D_IN))
    y = (X @ spec["w"] + spec["b"] > 0).astype(int)
    return X, y


def _xor(spec, rng, n):
    # 对角双团：非线性边界（线性不可分）但边界平滑，500 步在线可学
    X = rng.normal(size=(n, D_IN)) * 0.4
    y = rng.integers(0, 2, size=n)
    centers = np.array([[-1.5, 1.5], [1.5, -1.5]])
    X[:, :2] = centers[y] + rng.normal(scale=0.9, size=(n, 2))
    return X, y


def _wave(spec, rng, n):
    # 波浪边界：非线性（线性不可分）但平滑，仅前两维有信息
    X = rng.normal(size=(n, D_IN)) * 0.6
    y = (X[:, 0] > 1.1 * np.sin(1.6 * X[:, 1])).astype(int)
    return X, y


def _blobs3(spec, rng, n):
    y = rng.integers(0, 3, size=n)
    X = spec["centers"][y] + rng.normal(scale=1.0, size=(n, D_IN))
    return X, y


def _moons(spec, rng, n):
    t = rng.uniform(0, np.pi, size=n)
    y = rng.integers(0, 2, size=n)
    X = np.zeros((n, D_IN))
    r = 2.5
    X[y == 0, 0] = r * np.cos(t[y == 0])
    X[y == 0, 1] = r * np.sin(t[y == 0])
    X[y == 1, 0] = r * np.cos(t[y == 1] + np.pi) + 1.2
    X[y == 1, 1] = r * np.sin(t[y == 1] + np.pi) - 0.4
    X += rng.normal(scale=0.35, size=(n, D_IN))
    return X, y


GENS = {"linear": _linear, "xor": _xor, "wave": _wave,
        "blobs3": _blobs3, "moons": _moons}

PLAN = ["linear", "xor", "wave", "blobs3", "linear", "xor", "moons"]


def build_stream(seed=7, n_train=600, n_test=200):
    """返回 (train_blocks, test_sets, names)。

    train_blocks: list of (X, y) 按时间顺序到达的训练块
    test_sets:    list of (X, y) 每个概念的固定测试集（按首次出现顺序）
    names:        概念名列表（与 test_sets 对齐）
    """
    rng = np.random.default_rng(seed)
    train_blocks, test_sets, names = [], [], []
    spec_seed = {}
    for kind in PLAN:
        if kind not in spec_seed:
            spec_seed[kind] = 1000 + 31 * len(spec_seed) + seed
        spec = SPECS[kind](np.random.default_rng(spec_seed[kind]))
        Xtr, ytr = GENS[kind](spec, rng, n_train)
        train_blocks.append((Xtr, ytr))
        if kind not in names:
            names.append(kind)
            test_rng = np.random.default_rng(spec_seed[kind] * 7 + 1)
            test_sets.append(GENS[kind](spec, test_rng, n_test))
    return train_blocks, test_sets, names


def stream_length(train_blocks):
    return sum(len(y) for _, y in train_blocks)
