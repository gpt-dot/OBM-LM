"""OBM-LM —— 衔尾蛇梦网的语言模型化扩展。"""
import os as _os

# 可复现性：BLAS 线程数决定 matmul 的归约顺序，而本系统的边界检测/合同
# 授予/分裂判据全是阈值决策——浮点末位差异会经混沌放大成完全不同的事件
# 序列（实测：同种子、同代码，auto 线程与单线程运行出现不同的分裂/封存
# 序列与细胞分工；而同一线程设置内重复运行位级一致）。在 numpy 被导入前
# 固定线程数，保证"同代码 + 同命令 ⇒ 位级同结果"。用户已在环境中显式
# 设置则尊重之（setdefault 不覆盖）。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from obm_lm.cell_lm import MicroLM
from obm_lm.lm_mesh import OBMLM, OBMLMConfig
from obm_lm.langs import build_blocks, PLAN, TASKS

__all__ = ["MicroLM", "OBMLM", "OBMLMConfig", "build_blocks", "PLAN", "TASKS"]
