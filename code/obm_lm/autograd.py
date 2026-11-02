"""mini-autograd —— 纯 numpy 反向模式自动微分。

覆盖 OBM-LM 所需算子：add/mul（含广播）、matmul（2D）、bmm（批量矩阵乘）、
sum/mean、transpose、reshape、gather（embedding lookup）、softmax、
layernorm（末维）、gelu、cross_entropy（mean）。

约定：所有 matmul 操作数为 2D；批量矩阵乘一律走 bmm；广播由 add/mul 自动处理。
"""
import numpy as np

_EPS = 1e-12


def _sum_to(g, shape):
    """把梯度归约到目标形状（广播反传）。"""
    g = np.asarray(g, dtype=float)
    if g.shape == shape:
        return g
    if g.ndim == 0:                      # 标量梯度扩散到任意形状
        return np.full(shape, float(g))
    # 前导维度求和
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
        elif g.shape[i] != s:
            g = np.broadcast_to(g, shape)
    return g.reshape(shape) if g.shape != shape else g


class T:
    """自动微分张量。"""
    __slots__ = ("data", "grad", "_back", "_prev")

    def __init__(self, data, _children=(), _back=None):
        self.data = data
        self.grad = np.zeros_like(data)
        self._prev = _children
        self._back = _back

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def _accum(self, g):
        g = np.asarray(g, dtype=self.data.dtype)
        self.grad = self.grad + g

    # ---------- 算术（广播安全） ----------
    def __add__(self, other):
        other = other if isinstance(other, T) else T(np.asarray(other, self.data.dtype))
        out = T(self.data + other.data, (self, other),
                lambda g: (self._accum(_sum_to(g, self.data.shape)),
                           other._accum(_sum_to(g, other.data.shape))))
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, T) else T(np.asarray(other, self.data.dtype))
        out = T(self.data * other.data, (self, other),
                lambda g: (self._accum(_sum_to(g * other.data, self.data.shape)),
                           other._accum(_sum_to(g * self.data, other.data.shape))))
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        other = other if isinstance(other, T) else T(np.asarray(other, self.data.dtype))
        out = T(self.data / other.data, (self, other),
                lambda g: (self._accum(_sum_to(g / other.data, self.data.shape)),
                           other._accum(_sum_to(-g * self.data / (other.data * other.data),
                                                other.data.shape))))
        return out

    __rtruediv__ = __truediv__

    def __sub__(self, other):
        other = other if isinstance(other, T) else T(np.asarray(other, self.data.dtype))
        out = T(self.data - other.data, (self, other),
                lambda g: (self._accum(_sum_to(g, self.data.shape)),
                           other._accum(_sum_to(-g, other.data.shape))))
        return out

    # ---------- 形状 ----------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = T(self.data.reshape(shape), (self,),
                lambda g: self._accum(g.reshape(self.data.shape)))
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(range(self.data.ndim))[::-1]
        out = T(np.transpose(self.data, axes), (self,),
                lambda g: self._accum(np.transpose(g, np.argsort(axes))))
        return out

    def __getitem__(self, idx):
        """仅支持切片索引（因果 mask 用）。"""
        out = T(self.data[idx], (self,),
                lambda g: self._accum(_scatter_slice(self.data, idx, g)))
        return out

    # ---------- 归约 ----------
    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,),
                lambda g: self._accum(_sum_to(np.expand_dims(g, axis) if axis is not None
                                              and not keepdims else g, self.data.shape)))
        return out

    def mean(self, axis=None, keepdims=False):
        d = self.data
        if axis is None:
            n = d.size
            out = T(d.mean(), (self,),
                    lambda g: self._accum(np.full_like(d, g / n)))
            return out
        ax = axis if isinstance(axis, tuple) else (axis,)
        n = 1
        for a in ax:
            n *= d.shape[a]
        out = T(d.mean(axis=axis, keepdims=keepdims), (self,),
                lambda g: self._accum(_sum_to(np.expand_dims(g, ax) if not keepdims else g,
                                              d.shape) / n))
        return out

    # ---------- 矩阵 ----------
    def __matmul__(self, other):
        """仅 2D。"""
        A, B = self.data, (other.data if isinstance(other, T) else np.asarray(other))
        other = other if isinstance(other, T) else T(B)
        out = T(A @ B, (self, other),
                lambda g: (self._accum(g @ B.T), other._accum(A.T @ g)))
        return out


def bmm(A, B):
    """批量矩阵乘：A (B,M,K) @ B (B,K,N) -> (B,M,N)。"""
    out = T(np.matmul(A.data, B.data), (A, B),
            lambda g: (A._accum(np.matmul(g, np.swapaxes(B.data, -1, -2))),
                       B._accum(np.matmul(np.swapaxes(A.data, -1, -2), g))))
    return out


def _scatter_slice(data, idx, g):
    buf = np.zeros_like(data)
    buf[idx] = g
    return buf


def gather(emb, idx):
    """embedding lookup：emb (V,D)，idx (...) int -> (...,D)。"""
    out = T(emb.data[idx], (emb,),
            lambda g: emb._accum(_scatter_add(emb.data, idx, g)))
    return out


def _scatter_add(data, idx, g):
    buf = np.zeros_like(data)
    np.add.at(buf, idx, g)
    return buf


# ---------- 激活 / 归一化 / 损失 ----------
def softmax(x, axis=-1):
    z = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    y = e / (e.sum(axis=axis, keepdims=True) + _EPS)
    out = T(y, (x,),
            lambda g: x._accum(y * (g - (g * y).sum(axis=axis, keepdims=True))))
    return out


def layernorm(x, eps=1e-5):
    """沿末维 LayerNorm（无 affine，affine 由外部加权乘加实现）。"""
    d = x.data
    mu = d.mean(axis=-1, keepdims=True)
    xc = d - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    D = d.shape[-1]
    out = T(xc * inv, (x,),
            lambda g: x._accum(_ln_bw(g, xc, inv, D)))
    return out


def _ln_bw(g, xc, inv, D):
    """LayerNorm 反向（LN 标准推导）。"""
    gs = g.sum(axis=-1, keepdims=True)            # (...,1)
    gxs = (g * xc).sum(axis=-1, keepdims=True)    # (...,1)
    return (g - gs / D - xc * inv ** 2 * gxs / D) * inv


def gelu(x):
    """tanh 近似 GELU。"""
    d = x.data
    t = np.tanh(0.7978845608 * (d + 0.044715 * d ** 3))
    y = 0.5 * d * (1 + t)
    dy = 0.5 * (1 + t) + 0.5 * d * 0.7978845608 * (1 + 3 * 0.044715 * d ** 2) * (1 - t * t)
    out = T(y, (x,), lambda g: x._accum(g * dy))
    return out


def cross_entropy(logits, targets):
    """logits (...,V) / targets (...) int -> 标量（mean over positions）。"""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / (e.sum(axis=-1, keepdims=True) + _EPS)
    flat = p.reshape(-1, p.shape[-1])
    N = targets.size
    ll = -np.log(flat[np.arange(N), targets.reshape(-1)] + _EPS).mean()
    out = T(ll, (logits,),
            lambda g: logits._accum(g * (p - _onehot(targets, p.shape[-1])) / N))
    return out


def _onehot(idx, V):
    oh = np.zeros((idx.size, V), dtype=np.float64)
    oh[np.arange(idx.size), idx.reshape(-1)] = 1.0
    return oh.reshape(*idx.shape, V)


# ---------- 优化器 ----------
def adam(params, lr, state=None, b1=0.9, b2=0.999, eps=1e-8):
    state = state if state is not None else {}
    t = state.get("t", 0) + 1
    for i, p in enumerate(params):
        m = state.setdefault(f"m{i}", np.zeros_like(p.data))
        v = state.setdefault(f"v{i}", np.zeros_like(p.data))
        m[...] = b1 * m + (1 - b1) * p.grad
        v[...] = b2 * v + (1 - b2) * (p.grad * p.grad)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        p.data -= lr * mh / (np.sqrt(vh) + eps)
    state["t"] = t
    return state


def zero_grad(params):
    for p in params:
        p.grad = np.zeros_like(p.data)


def backward(out, seed=1.0):
    """从 out 出发拓扑反向；out.grad 若为零则播种 seed（默认 d(sum(out))）。"""
    if not np.any(out.grad):
        out.grad = np.full_like(out.data, seed) if out.data.shape else np.array(seed, dtype=out.data.dtype)
    topo, seen = [], set()
    def build(t):
        if id(t) in seen:
            return
        seen.add(id(t))
        for p in t._prev:
            build(p)
        topo.append(t)
    build(out)
    for t in reversed(topo):
        if t._back is None:
            continue
        g = t.grad
        g = np.asarray(g, dtype=t.data.dtype)
        if g.shape != t.data.shape:
            g = np.broadcast_to(g, t.data.shape)
        t._back(g)
