"""autograd 数值梯度验证：每个算子的解析梯度 vs 中心差分。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.autograd import (T, bmm, gather, softmax, layernorm, gelu,
                             cross_entropy, backward)

rng = np.random.default_rng(0)
fails = []


def num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + eps
        fp = float(f().data)
        x.data[i] = old - eps
        fm = float(f().data)
        x.data[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check(name, f, x):
    out = f()
    x.grad = np.zeros_like(x.data)
    out.grad = np.ones_like(out.data) if out.data.shape else 1.0
    backward(out)
    ana = x.grad.copy()
    ng = num_grad(f, x)
    err = float(np.max(np.abs(ana - ng)))
    ok = err < 1e-5
    print(f"  {name:24s} max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)


# 固定随机操作数，保证数值/解析梯度面对同一函数
a1 = T(rng.normal(size=(3, 4)))
b_b = T(rng.normal(size=(4,)))
check("add 广播", lambda: (a1 + b_b).sum(), a1)

b_c = T(rng.normal(size=(3, 1)))
check("mul 广播", lambda: (a1 * b_c).sum(), a1)

m1 = T(rng.normal(size=(3, 5)))
m2 = T(rng.normal(size=(5, 4)))
check("matmul", lambda: (m1 @ m2).sum(), m1)
check("matmul(B)", lambda: (m1 @ m2).sum(), m2)

b1 = T(rng.normal(size=(2, 3, 5)))
b2m = T(rng.normal(size=(2, 5, 4)))
check("bmm", lambda: bmm(b1, b2m).sum(), b1)
check("bmm(B)", lambda: bmm(b1, b2m).sum(), b2m)

s1 = T(rng.normal(size=(2, 3, 4)))
check("sum(axis=1)", lambda: s1.sum(axis=1).sum(), s1)
check("sum(keepdims)", lambda: s1.sum(axis=2, keepdims=True).sum(), s1)
check("mean(axis=0)", lambda: s1.mean(axis=0).sum(), s1)
check("transpose", lambda: s1.transpose(0, 2, 1).sum(), s1)
check("reshape", lambda: s1.reshape(4, 6).sum(), s1)

sm = T(rng.normal(size=(3, 5)) * 2)
check("softmax", lambda: softmax(sm, axis=-1).sum(), sm)

ln = T(rng.normal(size=(2, 3, 8)))
check("layernorm", lambda: layernorm(ln).sum(), ln)

gl = T(rng.normal(size=(3, 4)) * 1.5)
check("gelu", lambda: gelu(gl).sum(), gl)

emb = T(rng.normal(size=(10, 4)))
idx = rng.integers(0, 10, size=(3, 5))
check("gather", lambda: gather(emb, idx).sum(), emb)

lg = T(rng.normal(size=(4, 6)))
tg = rng.integers(0, 6, size=(4,))
check("cross_entropy", lambda: cross_entropy(lg, tg), lg)

# 组合：一个小前向（模拟模型的典型链路）
Wq = T(rng.normal(size=(12, 12)) * 0.3)
Wv = T(rng.normal(size=(12, 12)) * 0.3)
def mini_attn():
    h = layernorm(s1.reshape(2, 12))
    q = h @ Wq
    k = h @ Wv
    att = softmax((q @ k.transpose()) * 0.5, axis=-1)
    return (att @ h).sum()
check("组合 attn", mini_attn, s1)

print()
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("全部算子数值梯度验证通过")
