"""verify_gaussian_ctx.py —— 离线验证：解码 x 空间 + 每段对角高斯能否正确归属。

不跑模型，直接模拟封存窗口（每段取任务块中部的 200 个纯样本），
然后看各任务测试输入的 top-1 段归属。
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=1))

# 预计算伪逆（8x30）：p = concat(x@W0, x@W1, x@W2) = P x，P 为 30x8
S = np.vstack([W.T for W in model.sensory])   # 30x8
S_pinv = np.linalg.inv(S.T @ S) @ S.T     # 8x30
model.S_pinv = S_pinv

def decode(p):
    u = np.arctanh(np.clip(p, -1 + 1e-9, 1 - 1e-9))
    return S_pinv @ u

# 验证解码精度
X0 = train_blocks[0][0][:50]
err = max(np.abs(decode(model.perceive(x)) - x).max() for x in X0)
print(f"解码最大误差: {err:.2e}  （应 ~1e-15）")

PLAN = ["linear", "xor", "wave", "blobs3", "linear", "xor", "moons"]
# 模拟封存段：每个任务块取中部 200 样本；复现任务各自成段
seg_defs = []   # (seg_id, task_name, block_idx, [a,b])
sid = 0
seen = {}
for bi, name in enumerate(PLAN):
    if name not in seen:
        seen[name] = []
    seen[name].append(bi)
for name, bis in seen.items():
    for bi in bis:
        seg_defs.append((sid, name, bi))
        sid += 1

VAR_FLOOR = 0.05
def fit(bi, a=200, b=400):
    X = train_blocks[bi][0][a:b]
    xs = np.array([decode(model.perceive(x)) for x in X])
    m = xs.mean(0)
    v = np.maximum(xs.var(0), VAR_FLOOR)
    return m, v

segs = [(sid_, name, *fit(bi)) for sid_, name, bi in seg_defs]

def diag_ll(x, m, v):
    return -0.5 * np.sum(np.log(2 * np.pi * v) + (x - m) ** 2 / v)

print("\n== 每段拟合的逐维方差（解码空间） ==")
for sid_, name, m, v in segs:
    print(f"  段#{sid_} {name:8s} var={np.round(v, 2)}")

print("\n== top-1 归属（T=1.0） ==")
for ti, (X, y) in enumerate(test_sets):
    tname = names[ti]
    hit = 0
    dist = {}
    for x in X:
        xd = decode(model.perceive(x))
        lls = [(sid_, diag_ll(xd, m, v)) for sid_, nm, m, v in segs]
        best = max(lls, key=lambda t: t[1])[0]
        dist[best] = dist.get(best, 0) + 1
        want = [s[0] for s in segs if s[1] == tname]
        hit += int(best in want)
    print(f"  {tname:8s}: 命中 {hit}/{len(X)}  分布 {dist}")

print("\n== 同任务两段之间的 LL 差（linear-1 vs linear-2，应接近 0） ==")
for x in test_sets[0][0][:5]:
    xd = decode(model.perceive(x))
    lls = {nm: diag_ll(xd, m, v) for sid_, nm, m, v in segs}
    ks = [k for k in lls if k == "linear"]
    vals = [lls[k] for k in ks]
    print(f"  linear 段 LL: {[round(v,2) for v in vals]}  差={vals[0]-vals[1]:+.3f}")
