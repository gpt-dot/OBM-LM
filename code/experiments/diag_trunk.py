"""快架构前提诊断：共享 trunk 能否学好 + 薄 head 单域适配上限。

这是「功能能否成立」的决定性前置实验：
  1) 混合预训练 SharedTrunk（256x4）N 步，看 loss 轨迹能否远低于随机基线 9.7
  2) 冻结 trunk，单独在 en_lit 上专门训练薄 head（2 层）M 步，测单域 val acc
     —— 这就是快架构「薄 head 在固定共享表征上专门化」的天花板

若单域 head acc 能到 0.4+，则深预训练 + trunk 随 OBM 进化有望跨过封门 0.55，
快架构功能可行；若卡在 0.2 以下，则共享混合 trunk 的单域有效容量被稀释，
快架构在此规模下无法等价替代「每细胞独立专门」。

用法：python3 experiments/diag_trunk.py
"""
import os, sys, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))  # experiments (for run_scale_fast import)
from obm_lm.corpus import build_streams, CACHE
from obm_lm.tokenizer import BPETokenizer
from obm_lm.fast_cell import SharedTrunk, FastCell
from run_scale_fast import build_fast_cfg

OUT = "results_scale"
tok = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
streams = build_streams(CACHE)
keys = list(streams.keys())
rng = np.random.default_rng(0)
V = tok.vocab_size
ce = torch.nn.functional.cross_entropy

# ---------- 1) 深预训练 trunk ----------
trunk = SharedTrunk(rng, V, 256, 8, 4, 256)
opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=0.01)
t0 = time.time()
PSTEPS = 500
print(f"== trunk pretrain {PSTEPS} steps (V={V})", flush=True)
for stp in range(PSTEPS):
    key = keys[stp % len(keys)]
    seq, tgt, _ = streams[key].batch(24, 256)
    seq = torch.as_tensor(np.asarray(seq), dtype=torch.long)
    tgt = torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long)
    opt.zero_grad()
    h = trunk(seq, train=True)
    B, L, D = h.shape
    logits = (torch.nn.functional.layer_norm(h, (D,)) @ trunk.tok_emb.t()).reshape(-1, V)
    loss = ce(logits, tgt)
    loss.backward(); opt.step()
    if (stp + 1) % 50 == 0:
        print(f"  [trunk] {stp+1}/{PSTEPS} loss={loss.item():.2f} ({time.time()-t0:.0f}s)", flush=True)
torch.save(trunk.state_dict(), os.path.join(OUT, "trunk_pretrained.pth"))
print("== trunk saved -> results_scale/trunk_pretrained.pth", flush=True)

# ---------- 2) 冻结 trunk，单域 head 适配上限 ----------
trunk.requires_grad_(False)
cfg = build_fast_cfg("pilot", 0, V, 2, 4)
cell = FastCell(rng, V, trunk, 256, 8, 2, 256, 0, cfg=cfg)
# FastCell 的 trunk 经 list 容器持有、不注册为子模块，故 cell.parameters()
# 只含薄 head 参数（不含 trunk）。冻结 trunk 后训练这些即单域 head 适配。
hopt = torch.optim.AdamW(list(cell.parameters()), lr=1.2e-3)
HSTEPS = 250
print(f"== head adapt on en_lit {HSTEPS} steps (trunk frozen)", flush=True)
for stp in range(HSTEPS):
    seq, tgt, _ = streams["en_lit"].batch(24, 256)
    seq = torch.as_tensor(np.asarray(seq), dtype=torch.long)
    tgt = torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long)
    hopt.zero_grad()
    logits = cell.forward_torch(seq, train=True)     # trunk no_grad, head grad
    loss = ce(logits.reshape(-1, V), tgt)
    loss.backward(); hopt.step()
    if (stp + 1) % 50 == 0:
        with torch.no_grad():
            s, t, _ = streams["en_lit"].batch(8, 256)
            ss = torch.as_tensor(np.asarray(s), dtype=torch.long)
            lg = cell.forward_torch(ss, train=False)
            pred = lg.argmax(-1).reshape(-1)
            tt = torch.as_tensor(np.asarray(t).reshape(-1), dtype=torch.long)
            acc = float((pred == tt).float().mean())
        print(f"  [head] {stp+1}/{HSTEPS} loss={loss.item():.2f} en_lit_val_acc={acc:.3f}", flush=True)
print("== DONE", flush=True)
