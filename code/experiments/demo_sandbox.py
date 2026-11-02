"""demo_sandbox.py —— 沙箱推理 demo：加载权重-only 存档并生成文本。

用途：scnet 远端训练完成后，把 *_weights.pkl 拉回沙箱，用本脚本验证
"训完的模型在 8GB / CPU 沙箱里跑得起来"。

    python3 experiments/demo_sandbox.py --weights results_lab/scale2_weights.pkl \
        --prompt "def " --gen-len 48

注意：这是多细胞 OBM-LM，推理走 cond_reliability 加权的细胞集成（与
eval_ppl 同路由逻辑），不是单点 transformer。
"""
import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from obm_lm.lm_mesh import OBMLM
from obm_lm import backend as bk
from obm_lm.tokenizer import BPETokenizer

_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_OBM_ROOT, "data", "cache")


def generate(mesh, tok, prompt, gen_len=48, ctx=None):
    ctx = ctx or mesh.cfg.max_len
    # 关键修复：encode 返回 numpy 数组，必须先转 list，否则
    # `ids = [0] + ids` 走 numpy 的广播语义 -> 长度不增长 -> 死循环。
    ids = list(tok.encode(prompt))
    ids = ids[-(ctx - 1):]
    # 左补齐到 ctx（用 0 当 pad），便于首次前向对齐位置编码
    while len(ids) < ctx:
        ids = [0] + ids
    ids = ids[-ctx:]
    seq = np.array([ids], dtype=np.int64)
    t0 = time.time()
    for _ in range(gen_len):
        L = seq.shape[1]
        ctxd = mesh._context(mesh.cells[0].signature(seq))
        rels = np.array([c.cond_reliability(ctxd) for c in mesh.cells])
        sel, _ = mesh._select(rels, allow_reflex=False)
        w = np.exp(mesh.cfg.ensemble_beta * rels[sel])
        w = w / w.sum()
        logits = None
        for j, ci in enumerate(sel):
            lg = bk.forward(mesh.cells[ci], seq, train=False)  # (1,L,V) numpy
            if logits is None:
                logits = np.zeros_like(lg)
            logits = logits + w[j] * lg
        nxt = int(np.argmax(logits[0, -1]))
        ids = ids + [nxt]
        ids = ids[-ctx:]
        seq = np.array([ids], dtype=np.int64)
    return tok.decode(ids), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="权重-only 存档路径")
    ap.add_argument("--prompt", default="def ")
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--n", type=int, default=3, help="重复采样次数（看稳定性）")
    ap.add_argument("--max-params-gb", type=float, default=5.5,
                    help="沙箱内存预算(GB)；超限时按信誉保留最重要的细胞，保证 8GB 装得下")
    args = ap.parse_args()

    tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
    t = time.time()
    mesh = OBMLM.__new__(OBMLM)  # 不触发随机建细胞
    import pickle
    with open(args.weights, "rb") as f:
        d = pickle.load(f)
    # 用存档里的 cfg 建壳，再载入权重（按预算裁剪细胞以适配 8GB 沙箱）
    mesh.__init__(d["cfg"])
    mesh.load_weights_only(args.weights,
                          param_budget_bytes=int(args.max_params_gb * 1e9))
    load_sec = time.time() - t
    print(f"[load] cells={len(mesh.cells)} params={mesh.n_params/1e6:.1f}M "
          f"step={mesh.step} load={load_sec:.1f}s", flush=True)

    for i in range(args.n):
        text, sec = generate(mesh, tok, args.prompt, gen_len=args.gen_len)
        print(f"\n--- sample {i+1} ({sec:.1f}s) ---")
        print(text)


if __name__ == "__main__":
    main()
