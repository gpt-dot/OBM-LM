"""note 日志器：找出 copy 专才 cell #0 的 seg_rel(3)（shift 合同）来源。

假设待验证：switch 窗口内（封存前 seg_counter 仍指旧段），受授细胞在
新任务数据上被记账到旧段号；段复现时旧段号经合并转移，造成跨任务合同污染。
"""
import os, sys

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # 位级可复现：固定 BLAS 线程（见 obm_lm/__init__.py）

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm.langs import build_blocks

BATCH = 12
NOTES = []          # (step, cell_id, seg, ok, r_after)

from obm_lm.cell_lm import MicroLM
_orig_note = MicroLM.note


def note_spy(self, seg_id, ok, alpha):
    _orig_note(self, seg_id, ok, alpha)
    NOTES.append((CUR["step"], self.cell_id, seg_id, ok,
                  round(self.seg_rel[seg_id], 3)))


MicroLM.note = note_spy
CUR = {"step": 0}
_orig_run = OBMLM.run_step


def run_spy(self, seq, tgt, mask):
    CUR["step"] = self.step
    return _orig_run(self, seq, tgt, mask)


OBMLM.run_step = run_spy


def main():
    cfg = OBMLMConfig(seed=1)
    model = OBMLM(cfg)
    blocks, test_sets, names = build_blocks(0, 160)
    for bi, block in enumerate(blocks):
        for _ in range(600):
            seq, tgt, mask = block.batch(BATCH)
            model.run_step(seq, tgt, mask)
    print("=== 所有涉及 seg#3 的记账（cell, step）===")
    for st, cid, seg, ok, r in NOTES:
        if seg == 3:
            print(f"  step {st:4d} cell #{cid} seg#{seg} ok={ok} r={r}")
    print("\n=== cell #0 的全部记账 ===")
    for st, cid, seg, ok, r in NOTES:
        if cid == 0:
            print(f"  step {st:4d} cell #{cid} seg#{seg} ok={ok} r={r}")


if __name__ == "__main__":
    main()
