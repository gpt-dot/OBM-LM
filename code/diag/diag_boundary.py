"""diag_boundary.py —— wave 段内 3 次误判边界，是哪个信号触发的？"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obm import OuroborosMesh, OBMConfig
from obm.stream import build_stream

SEED = 1
train_blocks, test_sets, names = build_stream()
model = OuroborosMesh(OBMConfig(seed=SEED))

# 插桩：包一层 observe 前后状态
log = []
orig_seal = model._seal_segment
def seal_spy():
    log.append((model.step, "SEAL", f"ema={model.ema_acc:.3f} peak={model.peak_acc:.3f} "
                f"drop_run={model.drop_run} nbuf={len(model.buffer)}"))
    return orig_seal()
model._seal_segment = seal_spy

step = 0
for X, y in train_blocks:
    for x, yy in zip(X, y):
        rec = model.run_step(x, yy)
        if 1150 <= step <= 1800 and step % 10 == 0:
            log.append((step, "tick",
                        f"ema={model.ema_acc:.3f} peak={model.peak_acc:.3f} "
                        f"run={model.drop_run} acc={rec['correct']}"))
        step += 1

for st, typ, msg in log:
    mark = "  <<<" if typ == "SEAL" else ""
    print(f"[{st:5d}] {typ:5s} {msg}{mark}")
