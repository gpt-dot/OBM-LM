"""插桩：打印每次梦境的 ens_acc / 池构成 / in_dip，定位误分裂。"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.lm_mesh import OBMLM, OBMLMConfig

_orig_dream = OBMLM.dream


def dream_spy(self):
    pools = self._pools()
    eval_samples = []
    for _, samples in pools:
        eval_samples.extend(samples[:12])
    eval_samples = eval_samples[:96]
    seq = np.array([s for s, _ in eval_samples])
    tgt = np.array([t for _, t in eval_samples])
    ens_acc = self._ensemble_acc(seq, tgt)
    # 每个 chunk 的 ctx 与池
    details = []
    for b in range(0, len(seq), self.cfg.batch):
        chunk = seq[b:b + self.cfg.batch]
        t = tgt[b:b + self.cfg.batch]
        ctx = self._context(self.cells[0].signature(chunk))
        rels = np.array([c.cond_reliability(ctx) for c in self.cells])
        sel, _ = self._select(rels, allow_reflex=False)
        accs = []
        for c in self.cells:
            pred = np.argmax(c.forward(chunk).data, -1)
            from obm_lm.lm_mesh import _OUT_MASK
            accs.append(round(float(np.mean(pred[:, _OUT_MASK] == t[:, _OUT_MASK])), 2))
        details.append(f"ctx={{{', '.join(f'{k}:{v:.2f}' for k,v in sorted(ctx.items(), key=lambda kv:-kv[1])[:2])}}}"
                       f" sel={sel} accs={accs}")
    recent = [h["acc"] for h in self.history[-200:]]
    in_dip = bool(recent) and float(np.mean(recent)) < 0.7 * float(np.max(recent))
    if self.history:
        cur = self.history[-1]
        in_dip = in_dip or cur["acc"] < cur["ema"] - 0.15
    print(f"  [dream] step {self.step}: ens_acc={ens_acc:.3f} split_line=0.95 "
          f"in_dip={in_dip} cells={len(self.cells)}", flush=True)
    led = {c.cell_id: {k: round(v, 2) for k, v in c.seg_rel.items()}
           for c in self.cells}
    best = {c.cell_id: {k: round(v, 2) for k, v in c.seg_best.items()}
            for c in self.cells}
    print(f"      ledger={led}", flush=True)
    print(f"      seg_best={best}", flush=True)
    for d in details:
        print(f"      {d}", flush=True)
    _orig_dream(self)


OBMLM.dream = dream_spy

if __name__ == "__main__":
    from obm_lm.langs import build_blocks
    cfg = OBMLMConfig(seed=1)
    model = OBMLM(cfg)
    blocks, test_sets, names = build_blocks(0, 160)
    for bi, block in enumerate(blocks[:3]):                 # 前 3 段即可
        for _ in range(600):
            seq, tgt, mask = block.batch(cfg.batch)
            model.run_step(seq, tgt, mask)
    print("[done]")
