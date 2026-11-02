"""插桩诊断：记录合同记账流水，定位 shift 专才消失 + sort 只有 0.78 的根因。"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm.langs import TASKS, PLAN
from obm_lm.cell_lm import MicroLM

NOTE_LOG = []          # (step, cell_id, seg, ok, r_after, bar)
DEATH_LOG = []

_orig_note = MicroLM.note
def note_spy(self, seg_id, ok, alpha):
    r0 = self.seg_rel.get(seg_id, 0.5)
    _orig_note(self, seg_id, ok, alpha)
    NOTE_LOG.append((CUR["step"], self.cell_id, seg_id, ok,
                     round(self.seg_rel[seg_id], 3)))
MicroLM.note = note_spy

_orig_dream = OBMLM.dream
def dream_spy(self):
    _orig_dream(self)
    for c in self.cells:
        DEATH_LOG.append((self.step, c.cell_id,
                          {k: round(v, 2) for k, v in c.seg_rel.items()}))
OBMLM.dream = dream_spy

CUR = {"step": 0}
_orig_run = OBMLM.run_step
def run_spy(self, seq, tgt, mask):
    CUR["step"] = self.step
    d = _orig_run(self, seq, tgt, mask)
    if self.step % 100 == 0:
        led = {c.cell_id: {k: round(v, 2) for k, v in c.seg_rel.items()}
               for c in self.cells}
        print(f"  step {self.step:4d}: sel={d['selected']} "
              f"accs={[round(a,2) for a in d['cell_accs']]} ledger={led}",
              flush=True)
    if 1795 <= self.step <= 1960 and self.step % 5 == 0:
        ctx = d["ctx"]
        rels = d["rels"]
        print(f"    [dbg] step {self.step}: ctx={{{', '.join(f'{k}:{v:.2f}' for k,v in sorted(ctx.items(), key=lambda kv:-kv[1]))}}} "
              f"rels={[round(float(r),2) for r in rels]} sel={d['selected']} "
              f"accs={[round(a,2) for a in d['cell_accs']]}", flush=True)
    return d
OBMLM.run_step = run_spy


def main():
    seed = 1
    cfg = OBMLMConfig(seed=seed)
    model = OBMLM(cfg)
    from obm_lm.langs import build_blocks
    blocks, test_sets, names = build_blocks(0, 160)
    test_pairs = [(s, t) for s, t, _ in test_sets]
    masks = [m for _, _, m in test_sets]
    print("blocks:", [b.name for b in blocks], flush=True)
    for bi, block in enumerate(blocks[:4]):                 # 前 4 段
        for _ in range(600):
            seq, tgt, mask = block.batch(cfg.batch)
            model.run_step(seq, tgt, mask)
    accs = model.evaluate(test_pairs, masks)
    print("final per-task:", names, [round(a, 3) for a in accs], flush=True)
    print("\n=== 记账流水（shift 段 1800-2420，cell #7/#8/#9）===", flush=True)
    for rec in NOTE_LOG:
        st, cid, seg, ok, r = rec
        if 1800 <= st <= 2450 and cid in (7, 8, 9):
            print(f"  step {st:4d} cell #{cid} seg#{seg} ok={ok} r={r}", flush=True)
    print("\n=== 每次梦境的细胞账本 ===", flush=True)
    for st, cid, led in DEATH_LOG:
        print(f"  step {st:4d} cell #{cid}: {led}", flush=True)


if __name__ == "__main__":
    main()
