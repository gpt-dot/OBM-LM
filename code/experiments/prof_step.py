"""run_step 分相位性能剖析：定位 9.6s/step 的去向。"""
import sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(8)
from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm.corpus import DomainStream
from obm_lm.tokenizer import BPETokenizer

tok = BPETokenizer.load(os.path.join(_ROOT, "obm_mvp/data/cache/tokenizer.json"))
streams = {d: DomainStream(d, np.load(fos.path.join(_ROOT, "obm_mvp/data/cache/{d}_tokens.npy")), seed=1000+k)
           for k, d in enumerate(["en_lit", "zh_classic", "code_py", "en_wiki"])}

cfg = OBMLMConfig(seed=7, backend="torch", vocab=tok.vocab_size,
             d_model=256, n_heads=8, n_layers=4, max_len=256, batch=24,
             n_cells_init=4, max_cells=6, out_full=True, acc_prior=0.0,
             amp_bf16=True, grad_clip=1.0, weight_decay=0.01, lr=1.2e-3,
             lr_breach=4e-4, dream_lr=1.2e-3,
             fulfill_acc=0.12, expert_acc=0.30,
             award_reprise_d=3.0, award_reprise_probe=0.20,
             split_rel_frac=0.85, split_cooldown=400,
             dream_batches=8, dream_batch=16, buffer_size=600,
             seg_buffer_size=120, gauss_chi=0.0)
mesh = OBMLM(cfg)

BS, CTX = 24, 256
st = streams["en_lit"]
seq, tgt, mask = st.batch(BS, CTX)

# 预热
mesh.run_step(seq, tgt, mask)
mesh.step = 0

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
for i in range(3):
    seq, tgt, mask = st.batch(BS, CTX)
    mesh.run_step(seq, tgt, mask)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(35)
print(s.getvalue())
