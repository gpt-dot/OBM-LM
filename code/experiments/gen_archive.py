"""gen_archive.py —— 跨代生成可读性档案（TRIZ No.26 复制）。

用 8 条固定跨域 prompt 在 scale4/5/6 权重上生成，输出 JSON + HTML 对比报告。
评估只看 ppl 的矛盾由此补上"可读性"一翼：同一 prompt 三代并排，
肉眼即可判断"学舌 -> 仿写"的能力演进。

用法：python3 experiments/gen_archive.py
"""
import os, sys, json, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from obm_lm.lm_mesh import OBMLM
from obm_lm import backend as bk
from obm_lm.tokenizer import BPETokenizer
from demo_sandbox import generate

_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_OBM_ROOT, "data", "cache")

def _find_model(tag):
    """解压后自动寻找匹配的训后权重：models/ -> 包根 -> 上级 -> /workspace。"""
    for d in (os.path.join(_ROOT, "models"), _ROOT,
              os.path.dirname(_ROOT), "/workspace"):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if (f.startswith(f"OBM-LM-{tag}-") and f.endswith(".pkl")
                        and "训后" in f):
                    return os.path.join(d, f)
    return os.path.join(_ROOT, "models", f"OBM-LM-{tag}-训后-75M.pkl")

WEIGHTS = [(t, _find_model(t)) for t in ("scale4", "scale5", "scale6")]

PROMPTS = [
    ("en_lit",  "The old man looked at the sea and"),
    ("en_lit",  "It was a dark night in Paris, and"),
    ("zh_classic", "子曰：学而时习之，不亦"),
    ("zh_classic", "天地玄黄，宇宙"),
    ("code_py", "def quicksort(arr):"),
    ("code_py", "import numpy as np\n# compute the mean of"),
    ("en_wiki", "The Roman Empire was"),
    ("en_wiki", "Photosynthesis is the process by which"),
]

GEN_LEN = 40

# 按路由细胞自动选温（头脑风暴 #3 温度扫描实测）
AUTO_TEMP = {3: 0.7}   # zh/code 专才 -> 低温保正确性；其余 -> 0.9 换多样性


def generate_sampled(mesh, tok, prompt, gen_len=GEN_LEN, ctx=None,
                     temp=0.8, top_k=40, rep_pen=1.15, seed=0):
    """采样解码（TRIZ No.35/No.13）：温度 + top-k + 重复惩罚，打破贪心循环。"""
    rng = np.random.default_rng(seed)
    ctx = ctx or mesh.cfg.max_len
    ids = list(tok.encode(prompt))
    ids = ids[-(ctx - 1):]
    while len(ids) < ctx:
        ids = [0] + ids
    ids = ids[-ctx:]
    seq = np.array([ids], dtype=np.int64)
    t0 = time.time()
    for _ in range(gen_len):
        ctxd = mesh._context(mesh.cells[0].signature(seq))
        rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
        sel, _ = mesh._select(rels, allow_reflex=False)
        w = np.exp(mesh.cfg.ensemble_beta * rels[sel])
        w = w / w.sum()
        logits = None
        for j, ci in enumerate(sel):
            lg = bk.forward(mesh.cells[ci], seq, train=False)
            if logits is None:
                logits = np.zeros_like(lg)
            logits = logits + w[j] * lg
        z = logits[0, -1].astype(np.float64)
        # 重复惩罚：已出现 token 的 logits 向下压
        for t in set(ids):
            if z[t] > 0:
                z[t] /= rep_pen
            else:
                z[t] *= rep_pen
        z = z / temp
        if top_k and top_k < len(z):
            kth = np.sort(z)[-top_k]
            z[z < kth] = -1e9
        p = np.exp(z - z.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        ids = ids + [nxt]
        ids = ids[-ctx:]
        seq = np.array([ids], dtype=np.int64)
    return tok.decode(ids), time.time() - t0


def load_mesh(path, budget_gb=5.5):
    with open(path, "rb") as f:
        d = pickle.load(f)
    mesh = OBMLM.__new__(OBMLM)
    mesh.__init__(d["cfg"])
    mesh.load_weights_only(path, param_budget_bytes=int(budget_gb * 1e9))
    return mesh


def load_zoo(paths, budget_gb=5.5):
    """细胞动物园（头脑风暴 #2b）：多代权重的细胞合并进一个 mesh，
    NLL 路由按域挑 strongest —— 实测严格占优任何单代
    （en_lit 123 / zh 823 / code 81 / wiki 295）。"""
    paths = paths if isinstance(paths, list) else paths.split(",")
    meshes = [load_mesh(p.strip(), budget_gb) for p in paths]
    zoo = meshes[0]
    for m in meshes[1:]:
        # 注意：不同代各细胞独立编号，cell_id 撞号≠同一细胞——
        # 9 细胞 zoo 曾因按 cell_id 去重塌缩成 3（s5 的 code 专才被误删）。
        zoo.cells.extend(m.cells)
    return zoo


def prune_zoo(mesh, streams, factor=2.0, ctx=128, n_win=48, verbose=True):
    """细胞凋亡（SCAMPER-E × 免疫学"凋亡"类比）。

    园丁页实测：细胞#1 全域 ppl 差最优者 2-40×，是纯死重——
    不仅浪费内存，NLL 路由决策是 O(N_cells)，每多一个细胞全体慢一拍。
    判据：某细胞在**所有域**上都比该域最优细胞差 factor 倍以上 -> 凋亡移除。
    免疫学对应：未被任何抗原（域）需要的克隆走向 apoptosis，而非 tolerance。
    返回 (保留数, 凋亡明细)。"""
    import numpy as _np
    DOMS = list(streams.keys())
    M = _np.zeros((len(DOMS), len(mesh.cells)))
    for di, k in enumerate(DOMS):
        seq, tgt = streams[k].eval_windows(ctx=ctx)
        seq, tgt = seq[:n_win], tgt[:n_win]
        for ci, c in enumerate(mesh.cells):
            vals = [bk.mean_nll(c, seq[b:b + 12], tgt[b:b + 12])
                    for b in range(0, len(seq), 12)]
            M[di, ci] = _np.exp(_np.mean(vals))
    dead = []
    for ci in range(len(mesh.cells) - 1, -1, -1):  # 倒序删除，保留至少 1 个
        if len(mesh.cells) - len(dead) <= 1:
            break
        if all(M[di, ci] > factor * M[di].min() for di in range(len(DOMS))):
            dead.append(mesh.cells[ci].cell_id)
            del mesh.cells[ci]
    if verbose and dead:
        print(f"  [apoptosis] 移除细胞 {dead}（全域均 >{factor}x 领地最优）"
              f" -> {len(mesh.cells)} cells", flush=True)
    return len(mesh.cells), dead


def generate_nll(mesh, tok, prompt, gen_len=GEN_LEN, ctx=None,
                 temp=0.8, top_k=40, rep_pen=1.15, beta_nll=4.0, seed=0):
    """NLL 路由 + 采样解码（TRIZ No.24 中介物）。

    用 prompt 上的逐细胞平均 NLL 做软路由权重（模型自身似然当中介信号，
    绕过合同市场——实测市场路由漏用专才细胞：code 专才 87 vs 市场 284）。
    """
    rng = np.random.default_rng(seed)
    ctx = ctx or mesh.cfg.max_len
    ids = list(tok.encode(prompt))
    ids = ids[-(ctx - 1):]
    while len(ids) < ctx:
        ids = [0] + ids
    ids = ids[-ctx:]
    # 1) 逐细胞在 prompt 上的 teacher-forced NLL -> 路由权重
    seq0 = np.array([ids], dtype=np.int64)
    tgt0 = np.array([ids[1:] + [0]], dtype=np.int64)
    nlls = []
    for c in mesh.cells:
        nlls.append(bk.mean_nll(c, seq0, tgt0))
    nlls = np.array(nlls)
    w_all = np.exp(-beta_nll * (nlls - nlls.min()))
    w_all = w_all / w_all.sum()
    sel = list(range(len(mesh.cells)))
    t0 = time.time()
    for _ in range(gen_len):
        logits = None
        for j, ci in enumerate(sel):
            lg = bk.forward(mesh.cells[ci], seq0, train=False)
            if logits is None:
                logits = np.zeros_like(lg)
            logits = logits + w_all[j] * lg
        z = logits[0, -1].astype(np.float64)
        for t in set(ids):
            if z[t] > 0:
                z[t] /= rep_pen
            else:
                z[t] *= rep_pen
        z = z / temp
        if top_k and top_k < len(z):
            kth = np.sort(z)[-top_k]
            z[z < kth] = -1e9
        p = np.exp(z - z.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        ids = ids + [nxt]
        ids = ids[-ctx:]
        seq0 = np.array([ids], dtype=np.int64)
    return tok.decode(ids), time.time() - t0, {
        mesh.cells[i].cell_id: round(float(n), 2) for i, n in enumerate(nlls)}


def generate_top1(mesh, tok, prompt, gen_len=GEN_LEN, ctx=None,
                  temp=None, top_k=40, rep_pen=1.15, n_pref=64,
                  tau=0.15, beta_nll=4.0, seed=0):
    """v3.1 推理栈：自适应路由 + 按细胞自动选温（头脑风暴 #3）。

    temp=None 时按路由结果自动选温（实测最优配方）：
    cell#3（zh/code 专才）-> 0.7（结构重复是特性，低温保正确性）；
    其他（en 散文）-> 0.9（高温换多样性）。

    路由决策（No.17 空间维数变化 + No.10 预先作用 + No.5 动态化）：
    prompt 期用**无填充**的真实尾部 n_pref token 做一次性 NLL 决策
    （左侧 [0] 填充会污染 NLL，实测全路由到同一细胞）：
    - 自信（top1-top2 差 >= tau）：纯 top1 单细胞，算力 1/N；
    - 模糊（差 < tau）：按 softmax 混入第二名（实测 zh ppl 791 vs 829）。
    """
    rng = np.random.default_rng(seed)
    ctx = ctx or mesh.cfg.max_len
    real = list(tok.encode(prompt))
    p = real[-n_pref:] if len(real) >= 2 else real
    w = np.ones(len(mesh.cells)) / max(len(mesh.cells), 1)
    if len(p) >= 2:
        s = np.array([p], dtype=np.int64)
        t = np.array([p[1:] + [0]], dtype=np.int64)
        nlls = np.array([bk.mean_nll(c, s, t) for c in mesh.cells])
        o = np.argsort(nlls)
        w = np.zeros(len(mesh.cells)); w[o[0]] = 1.0
        if nlls[o[1]] - nlls[o[0]] < tau:
            w[o[1]] = np.exp(-beta_nll * (nlls[o[1]] - nlls[o[0]]))
            w /= w.sum()
    sel = [i for i in range(len(mesh.cells)) if w[i] > 0.005]
    ws = [float(w[i]) for i in sel]
    if temp is None:
        # 按路由细胞自动选温（头脑风暴 #3 实测配方：zh/code 0.7、en 0.9）
        temp = AUTO_TEMP.get(mesh.cells[int(np.argmax(w))].cell_id, 0.9)
    ids = real[-(ctx - 1):]
    while len(ids) < ctx:
        ids = [0] + ids
    ids = ids[-ctx:]
    seq = np.array([ids], dtype=np.int64)
    t0 = time.time()
    for _ in range(gen_len):
        logits = None
        for j, ci in enumerate(sel):
            lg = bk.forward(mesh.cells[ci], seq, train=False)
            if logits is None:
                logits = np.zeros_like(lg)
            logits = logits + ws[j] * lg
        z = logits[0, -1].astype(np.float64)
        for t in set(seq[0].tolist()):
            z[t] = z[t] / rep_pen if z[t] > 0 else z[t] * rep_pen
        z = z / temp
        if top_k and top_k < len(z):
            z[z < np.sort(z)[-top_k]] = -1e9
        pr = np.exp(z - z.max())
        pr /= pr.sum()
        nxt = int(rng.choice(len(pr), p=pr))
        ids = (seq[0].tolist() + [nxt])[-ctx:]
        seq = np.array([ids], dtype=np.int64)
    text = tok.decode(seq[0].tolist())
    gen_text = text[len(prompt):] if text.startswith(prompt) else text
    route = {mesh.cells[i].cell_id: round(float(w[i]), 2)
             for i in range(len(mesh.cells)) if w[i] > 0.005}
    return gen_text, time.time() - t0, route


def main():
    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    out = {"gen_len": GEN_LEN, "prompts": [p for _, p in PROMPTS], "runs": []}
    for tag, path in WEIGHTS:
        if not os.path.exists(path):
            print(f"[skip] {path} missing"); continue
        t0 = time.time()
        mesh = load_mesh(path)
        print(f"[{tag}] cells={len(mesh.cells)} params={mesh.n_params/1e6:.1f}M "
              f"step={mesh.step} load={time.time()-t0:.1f}s", flush=True)
        run = {"tag": tag, "cells": len(mesh.cells),
               "params_m": round(mesh.n_params/1e6, 1), "gens": []}
        for dom, prompt in PROMPTS:
            text, sec = generate(mesh, tok, prompt, gen_len=GEN_LEN)
            gen_text = text[len(prompt):] if text.startswith(prompt) else text
            run["gens"].append({"domain": dom, "prompt": prompt,
                                "text": gen_text, "sec": round(sec, 1)})
            print(f"  [{dom}] greedy {sec:.1f}s -> {gen_text[:50]!r}", flush=True)
        # 采样解码对照（TRIZ No.35/No.13：改解码条件，不动模型）
        run["gens_sampled"] = []
        for k, (dom, prompt) in enumerate(PROMPTS):
            text, sec = generate_sampled(mesh, tok, prompt, seed=k)
            gen_text = text[len(prompt):] if text.startswith(prompt) else text
            run["gens_sampled"].append({"domain": dom, "prompt": prompt,
                                        "text": gen_text, "sec": round(sec, 1)})
            print(f"  [{dom}] sample {sec:.1f}s -> {gen_text[:50]!r}", flush=True)
        out["runs"].append(run)
    with open(os.path.join(_OBM_ROOT, "results_lab", "gen_archive.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[done] -> results_lab/gen_archive.json", flush=True)


if __name__ == "__main__":
    main()
