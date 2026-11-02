# 快训架构：共享 trunk + 薄专才 head（OBM-LM 加速方案）

> 目标：在**不改变 OBM-LM 性质**（无中央路由器、概念签名 + 条件合同市场驱动的
> 多细胞涌现分工）的前提下，把**训练算力**与**推理延迟**降下来，使规模（LLM
> 量级参数）与效果（专长分工、抗遗忘）保持不变。

---

## 1. 问题定位（TRIZ 技术矛盾）

原 OBM-LM 由 **N 个完全独立的全量 transformer 细胞**组成。算力随规模爆炸的两
大来源：

| 阶段 | 原架构成本 | 随规模的增长 |
|---|---|---|
| 训练（每步） | 1 次全量前向+反向（受授细胞）+ 每 `dist_refresh` 步 **N× 全量前向**刷新熵缓存 | N× 全量前向 |
| 推理（每 token） | **N× 全量前向**（全库锚点）+ k× 全量前向（集成） | N× 全量前向 |

> 核心矛盾：**「规模」与「专才训练」被绑死在同一份全量参数上**。细胞越多、
> 模型越大，成本同时被 N 和「全量」两个因子放大。

**TRIZ 矛盾陈述**：想要「#25 时间损失↓、#39 生产率↑」，但不想牺牲
「#1 规模↓、#27 可靠性/效果↓」。

| 想提升的参数 | 不想要的副作用 | 采用的 TRIZ 原理 |
|---|---|---|
| 时间损失↓ / 生产率↑ | 规模↓ / 效果↓ | **分割**（trunk/head 解耦）、**抽取**（不变表征抽成冻结 trunk）、**预先作用**（trunk 预训练一次）、**局部质量**（每细胞只看自己域）、**动态化**（head 随分裂/凋亡增减）、**替代**（签名路由替代全细胞枚举） |

---

## 2. 头脑风暴候选架构

| 候选 | 思路 | 保留「无中央路由器」涌现？ | 算力 | 判定 |
|---|---|---|---|---|
| **A. 共享 trunk + 薄专才 head** | 一个 LLM 量级 trunk 所有细胞共用，仅每细胞训 1~2 层薄 head | ✅ 路由仍是签名+合同市场 | 训练 ~N×↓、推理 ~N×↓ | **采用** |
| B. MoE（冻结专家 + 学习路由器） | 标准 MoE | ❌ 显式中央 router，破坏涌现属性 | 中 | 否决 |
| C. 共享底座 + 每细胞 LoRA | 只训 LoRA 适配器 | ✅ | 训练 ~N×↓ | 被 A 吸收（薄 head ≈ 输出级 LoRA+非线性） |
| D. 蒸馏大模型 | 训一次再蒸馏 | ❌ 失去细胞涌现分工 | 高 | 否决 |
| E. 梯度检查点 | 省显存不省算力 | ✅ | ~1× | 仅辅助 |
| F. int4/int8 训练 | 量化降算力 | ✅ | 2~4× | 小模型精度受损，备选 |
| G. 异步多进程并行 N 细胞 | 并行化 | ✅ | 受单 CPU 瓶颈 | 复杂度高 |
| H. LoRA 化 head（rank-r） | 比 2 层 head 更薄 | ✅ | 更省 | A 的细化方向 |

**选定 A**：唯一既保「无中央路由器」涌现属性、又获得 ~N× 训练/推理加速的方案。
薄 head 即「输出级 LoRA + 非线性」，是 LoRA（C/H）的超集，无需单独实现。

---

## 3. 架构设计（`obm_lm/fast_cell.py`）

```
            ┌─────────────── 共享 SharedTrunk（规模所在，d×L 可到 LLM 量级）───────────────┐
   token ──►│  tok_emb → L 层 transformer → LayerNorm → h(D)                              │
            └───────────┬───────────────────────────────────────────────────────────────┘
                        │  h 共享（每步只算 1 次，无 grad 传给 trunk）
        ┌───────────────┼───────────────────┬───────────────────┐
      cell#0 head     cell#1 head         cell#2 head         cell#3 head   （薄：1~2 层 + LM 头）
        │ 专精 en_lit    │ 专精 zh_classic    │ 专精 code_py        │ 专精 en_wiki
        ▼                ▼                    ▼                    ▼
   签名/合同市场/分裂/凋亡/梦境   —— 全部跑在 head 上，mesh（OBMLM）零改动复用
```

- **SharedTrunk**：规模所在（d=256→768, L=4→10），所有细胞共用，前向只算一次；
  预训练后默认冻结（或保留极小 `trunk_lr` 联合训练）。
- **FastCell**：继承 `TorchMicroLM` 全部接口（签名/合同/分裂/`state`/`load_state`
  等），只把"身体"从全量 transformer 换成 `trunk + 薄 head`。trunk 经 `list`
  容器持有（不注册为子模块），故 `n_params/parameters/optimizer` 仅含 head ——
  mesh 对细胞的账本、授予、梦境逻辑**完全不变**。
- **mesh 零改动**：`OBMLM_Fast` 仅替换细胞工厂，所有 OBM 机制原样运行。

---

## 4. 速度来源（同性质、低算力、快）

| 维度 | 原架构 | 快架构 | 加速 |
|---|---|---|---|
| 训练/步 | 1× 全量前向+反向 + 每 N 步 N× 全量前向 | 1× trunk 前向（冻结/共享）+ 1× 薄 head 前向/反向 | **~N×** |
| 推理/token | N× 全量前向 + k× 全量前向 | 1× trunk 前向 + k(≤3)× 薄 head 前向 | **~N×** |
| 参数（pilot 口径） | 25.5M（4× 6.4M） | 6.4M(trunk) + 4×1.05M(head) ≈ **10.6M** | ~2.4× 少 |
| 规模上限 | 受 8GB cgroup：每细胞全量+优化器 4×+深拷贝快照 → 易 OOM | trunk 共享不复制、head 极小 → 内存恒定 | 更稳 |

性质保留：概念签名（trunk 输出均值池化，共享下几何一致）、条件合同市场、
双信号边界、单份+复现授予、梦境代谢、记账存疑 —— **全部在 head 上运行**。

---

## 5. 验证状态

| 实验 | 配置 | 结果 |
|---|---|---|
| `fast_verify` | 冻结 trunk，60-step 预训练 | ✅ 链路不崩，6 块 OBM+eval 跑通；step_time 1.56/1.00s；参数 8.47M；事件未点火（trunk 近随机，预期内）；采样函数 `.max()` bug 已修 |
| `fast_pilot2` | **trunk-lr 1.5e-4，300-step 预训练，head 2 层** | 🔄 运行中（对标原 pilot_v1 基线） |

**关键验证点**（待 `fast_pilot2` 出数）：
1. 封存/分裂/凋亡/复现授予能否点火（证明涌现分工）；
2. 终局 ppl 是否对标原 pilot_v1（en_lit≈825 / zh≈2363 / code_py≈98k / en_wiki≈49k）；
3. step_time 是否显著低于原 S1（~1.86s/步）且**不随细胞数增长**。

---

## 6. 已知风险与缓解

1. **冻结 trunk 限制 head 专精** → 已加 `trunk-lr`：trunk 以极小 lr 在 OBM 阶段
   **联合训练**（仍共享、每步只算 1 次，速度优势保住）。薄 head 站在"持续进化
   的共享表征"上，更易跨过封门（0.55）。
2. **路由依赖签名（trunk 输出均值池化）** → trunk 共享下所有细胞用同一份表征，
   签名几何天然一致，跨域距离远大于同域，路由判别性不降。
3. **会话压缩会杀后台长进程** → 已加：① 每 eval/每块**增量落盘** `metrics.json`；
   ② `--resume <ckpt>` 断点续训（按 `done_blocks` 跳过已完成块）。
4. **trunk 联合训练引发轻微跨域干扰** → 正常多任务骨干行为，由 head 吸收域特异性；
   必要时可降 `trunk-lr` 或回退纯冻结。

---

## 7. 运行方式

```bash
# 冻结 trunk 基线（纯薄 head，最快，验证链路）
python3 experiments/run_scale_fast.py --scale pilot --trunk-layers 4 \
    --head-layers 1 --pretrain 60 --steps 40 --tag _verify --seed 0

# trunk 联合训练（推荐，对标原 pilot）
python3 experiments/run_scale_fast.py --scale pilot --trunk-layers 4 \
    --head-layers 2 --pretrain 300 --trunk-lr 1.5e-4 --steps 250 \
    --tag _pilot2 --seed 0

# 被压缩杀进程后断点续训
python3 experiments/run_scale_fast.py --scale pilot --trunk-layers 4 \
    --head-layers 2 --pretrain 300 --trunk-lr 1.5e-4 --steps 250 \
    --tag _pilot2 --seed 0 --resume results_scale/fast_pilot2/ckpt_latest.pkl
```
