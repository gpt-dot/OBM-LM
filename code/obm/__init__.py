"""OBM（Ouroboros Mesh，衔尾蛇梦网）—— 最小可行产品（MVP）。
基于 TRIZ × 头脑风暴概念设计的可运行实现。

设计文档中的五大机制在本包中的落地方式：
  M1 分形细胞   -> cell.MicroCell：所有细胞同构的微型 MLP（一层隐藏）
  M2 分歧路由   -> router.divergence_route：按"细胞群互相冲突程度"选细胞
  M3 合同市场   -> mesh.OuroborosMesh：可靠性记账 + 惰性衰减 + 集成加权
  M4 梦境代谢   -> mesh.OuroborosMesh.dream：反刍/反事实/蒸馏/凋亡/分裂
  M5 双向推理   -> MVP 中降级为"细胞自带局部梯度 + 溯因日志"（见 README 边界说明）
"""
from .cell import MicroCell
from .mesh import OuroborosMesh, OBMConfig
from . import router, stream, baselines

__all__ = ["MicroCell", "OuroborosMesh", "OBMConfig", "router", "stream", "baselines"]
__version__ = "0.1.0-mvp"
