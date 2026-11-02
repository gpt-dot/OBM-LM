"""lm_mesh.py —— OuroborosMesh 的语言模型化：OBM-LM 主循环。

从 MVP 的 mesh.py 逐机制移植（序列数据版）：

  领地识别   : 概念签名 = 冻结投影的 token 均值池化（B, D；mesh 级共享、
               不训练——MVP 的教训：训练中的 embedding 漂移会让段内
               距离失控，实测 2.9→32）；每段封存时保存在签名空间的对角
               高斯（gmean/gvar），上下文 = 输入签名到各段均值的马氏
               距离 softmax（高斯 LL 的 log-方差项会被低方差任务绑架，
               已退役——详见 _context docstring）
  条件合同   : seg_rel + cond_reliability(ctx) + 惰性衰减 + 封存冻结
                + note-on-current 记账（与 MVP 逐行同构；三处 LM 适配：
                未记账段先验 0.5 兜底防冒充、记账门限随资历上升、
                承诺度按峰值 seg_best 判定防瞬时记账污染误放专才）
  边界检测   : 双信号 OR——A 输出段精度 EMA 跌落且持续；B 签名空间分布
                距离（EW vs 段内冻结参照）超限且持续；段尾按首次超限截断
  段级合并   : 任务复现（签名高斯对称距离 ≤1.5 判定，0.7/0.3 折叠 + 合同迁移；
                余弦判据扛不住任务间 token 重叠，已退役）
  合同授予   : 无人过线时（新任务）授予单个"最自由候选细胞"（承诺度低者优先，
                probe 精度择优）——防专才被拉去培训异任务、防全池违约训练；
                **任期锁**：授予黏着到毕业/无进展/超时——切换窗口内不重抽
                彩票（轮作授予以所有人都学不会当前任务，且储备细胞被违约
                记账 eroded 到凋亡线以下，容量枯竭后新任务只能靠 probe
                彩票 cannibalize 专才：实测 copy 专才 534 步 shift 全额
                lr 训练，测试 0.94→0.24 雪崩）；**复现授予**：本 batch
                签名贴近某个已封存段（原始马氏距离 ≤3.0，同任务训练批次
                ≈1.8±1.0；不能用 ctx 权重——softmax 对异任务批次也饱和，
                实测 shift→copy 距离 5.08 仍拿 0.95 权重）、该段有专才
                （seg_rel ≥ specialist_line）且专才在本 batch 上真有本事
                （probe ≥0.5：跨任务精度实测 ≤0.36、同任务 ≥0.90，距离
                尾部漏网的异任务专才由此挡驾）时，合同直接授予专才，任期
                封顶 100 步——混淆窗口期"从零学旧任务的自由细胞"此前能
                赢 probe 彩票（实测 seed 2 复现在线恢复仅 0.596，而该段
                专才存量精度 0.97），复现不是新任务，没有理由不让存量
                知识立即上岗
  梦境代谢   : 专才加权反刍（细胞只梦见自己的领地：自我巩固而非抹平均）
                + 反事实（embedding 噪声）+ 分裂 + 凋亡（只剪从未建立过
                强专才的细胞，且不剪当前段在学细胞）
  快慢双系统 : 反射弧（单细胞直达）+ 分歧深思（最多 k_max 细胞）
  三因子学习 : 学习率门限宽松（fulfill_acc，允许突破/恢复）；记账门限严格
                （_note_bar，随资历从 fulfill_acc 升到 expert_acc）；
                记账另有**存疑豁免**——本 batch 输出段精度相对峰值深度
                跌落、边界嫌疑已持续、或 ctx 先验不指向当前段（分布距离
                信号依赖签名可分性，copy↔shift 的 token 范围重叠使其
                永不点火，但 ctx 直接读先验可识别）时数据的段归属不可信，
                不订合同（切换窗口内旧专才免于被异任务精度灌水：
                实测 copy 复现首步把 0.99 记进 shift 段账，seg_rel 造假
                到 0.55 堂堂毕业，该细胞的梦从此混入 38% shift 自污染）

时间单位统一为"步"（一步 = 一个 batch = 12 条序列）。
"""
import os
import numpy as np
from dataclasses import dataclass

from obm_lm.cell_lm import MicroLM
from obm_lm.langs import V, S_LEN, SEQ_LEN
from obm_lm.autograd import cross_entropy, backward, adam, zero_grad
from obm_lm import backend as bk
from obm.router import divergence_route

_EPS = 1e-12
# LM 输出段 mask：SEP 之后的位置（预测 out 的每个 token）；
# 输入段的恒等复制是平凡任务，不计入任务指标。
_OUT_MASK = np.zeros(SEQ_LEN, dtype=bool)
_OUT_MASK[S_LEN + 2:] = True


@dataclass
class OBMLMConfig:
    # 细胞
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 16
    n_cells_init: int = 4            # 4 细胞起步：4 个任务各有一个"自由细胞"
    #                                 可承接首份合同（首期实验 3 细胞时第 4 个
    #                                 任务不得不占用已有专才，牺牲其旧知识）
    max_cells: int = 8
    k_max: int = 3
    # 训练
    batch: int = 12
    lr: float = 0.006               # 履约细胞学习率（Adam）
    lr_breach: float = 0.002        # 违约细胞（约 1/3；实测 lr/7 会让难任务
    #                                 永远跨不过履约线——copy 相变需要全额 lr）
    fulfill_acc: float = 0.30       # 履约线：batch 输出段精度（随机水平 1/16 的 5 倍）
    dream_lr: float = 0.006           # 梦境学习率（MVP 0.05 的等比缩放：
    #                                 LM 参数量 70x、lr 1/8；MVP 经验 dream_lr
    #                                 必须接近训练 lr 才扛得住覆盖，实测 0.003
    #                                 太弱——最终评估比 MVP 差 ~50 倍）
    dream_batches: int = 16
    dream_batch: int = 24
    dream_period: int = 200
    buffer_size: int = 4800         # 当前段经历缓冲（条序列；12 条/步 = 400 步）
    seg_buffer_size: int = 120      # 封存段保留的高意外度片段数
    max_segments: int = 12
    # 合同
    reliability_alpha: float = 0.10
    dream_alpha: float = 0.05
    inactivity_decay: float = 0.9995
    contract_floor: float = 0.55
    # 相对记账线：batch 精度 ≥ 段内全库峰值 × note_peak_frac 也记功。
    # 绝对门限（_note_bar，fulfill→expert）是按玩具任务的精度尺度定的，
    # 在真实 LM 的难域不可达——pilot 实测 en_wiki 最强细胞 acc=0.056、
    # code_py 0.106，都低于 junior 线 0.12，note(ok=False) 把专才的
    # seg_rel 磨到 0.0，合同永远不成、路由退化成探针彩票（通才通吃，
    # code_py 路由 ppl 97861 vs 专才 1332，73× 差距）。相对线让"本段
    # 最强的细胞"能积累信誉——专才的意义本来就是相对分工，不是绝对精度。
    note_peak_frac: float = 0.8
    # 单份授予任期锁：合同授予后黏着到"毕业/无进展/超时"为止。
    # 授予原本是每步重抽的 probe 彩票（池空时连专才一起抽）——切换
    # 窗口（封存前 ~20 步 + 封存后合同池建立前）内受授细胞每步都可能
    # 易主。实测后果（seed 1）：copy 段 4 个细胞被轮番授予、全部违约
    # 记账，储备细胞 seg_rel eroded 到凋亡线以下被清剪（step 1400）；
    # 第 4 个任务到达时已无自由细胞，授予退化为专才彩票——copy 专才
    # #0 中标后 breach-lr 训练让其 shift 精度爬过履约线，ok 记账使其
    # seg_rel 涨过合同线"堂堂正正毕业"，随后 534 步全额 lr 的 shift
    # 训练把 copy 电路覆盖（测试 0.94→0.24 雪崩）。
    # 任期锁给受授细胞一段持续窗口把新任务推过合同线：专才不被抽签
    # 拖去培训异任务，自由储备也因始终空手而受剪枝保护
    # （`not c.seg_rel` 豁免），容量不再枯竭。
    award_max_tenure: int = 200      # 任期上限：超时释放重授（防永久占座）
    award_min_tenure: int = 100      # 最短任期：公平学习窗口（难任务慢热）
    award_stall_acc: float = 0.15    # 任期内最佳 batch 精度低于此 = 无进展
    award_reprise_d: float = 3.0     # 复现授予距离门限（批次签名到段高斯的原始
    #                                 马氏距离）。同任务训练批次实测 ≈1.8±1.0
    #                                 （段库存样本回放），最近跨任务对
    #                                 shift→reverse ≈4.5±2.7（尾部可低至 ~2.8，
    #                                 单靠距离分不干净——真正的判别在精度门限）
    award_reprise_probe: float = 0.50  # 复现授予的专才当批精度门限。语义是"存量
    #                                 知识直接上岗"：实测各专才跨任务精度 ≤0.36、
    #                                 同任务 ≥0.90，0.5 落在分离带内——距离尾部
    #                                 偶发命中的异任务专才（shift 批次落进
    #                                 reverse 段门限内）被这道门限挡在门外
    award_reprise_tenure: int = 100   # 复现授予的最长任期：超过即释放重授
    #                                 （正常窗口 ~22 步内由合同池接管；此为兜底——
    #                                 若边界检测始终不点火，释放后自由细胞承接
    #                                 导致精度跌落，反而触发边界检测把系统拉回正轨）
    specialist_line: float = 0.80     # 承诺度线：seg_best 峰值超过它的细胞视为
    #                                 "有存量知识的专才"，新合同优先授予自由细胞。
    #                                 按峰值而非当前 seg_rel 判定——切换窗口内
    #                                 3 个违约记账就能把 0.9 砍到 0.66，按当前值
    #                                 判定会把专才误放给异乡合同（实测 copy 专才
    #                                 被转化成 reverse 专才）
    commitment_decay: float = 0.999    # 承诺度峰值衰减（/步）：远慢于可靠性衰减
    #                                 （0.9995）——扛 ~50 步记账污染窗口，
    #                                 ~250 步异段缺席后释放细胞供复兴
    expert_acc: float = 0.65          # 记账的资历门限：seg_rel 从 0.5 爬到 0.9，
    #                                 履约标准从 fulfill_acc 线性升到 expert_acc
    #                                 （资历越深标准越高，防止"半吊子"细胞
    #                                 靠 0.30 的低线把领地可靠性刷到 1.0）
    ensemble_beta: float = 10.0
    reflex_conf: float = 0.85       # 反射弧：输出段 top-1 概率均值
    reflex_rel: float = 0.75
    # 领地匹配（签名空间）
    ctx_temp: float = 0.3
    ctx_uniform: float = 0.05
    var_floor: float = 0.05
    ctx_min_seg_n: int = 24
    ctx_freeze_n: int = 240          # 段内冻结参照的样本数（20 步）
    ew_alpha: float = 0.008          # 签名 EW 速率（每样本；一个 batch 12 条）
    d_thr: float = 3.0               # 分布距离阈值（冻结签名空间已标定：
    #                                 段内噪声 ≈0.5（极值 1.6），任务切换 ≥4.7）
    d_sustain: int = 15             # 按步计
    gauss_chi: float = 250.0        # 签名马氏离群剔除（64 维；样本方差估计
    #                                 使比值分布比 χ² 更厚尾，109 会砍掉一半）
    # 段级合并
    seg_merge_dist: float = 1.5      # 签名高斯的对称距离阈值（同任务复现 ≈0.3，
    #                                 跨任务 ≥3.8；余弦判据扛不住 token 范围重叠）
    buf_trim: float = 0.35
    # 边界检测 A
    acc_drop: float = 0.20
    drop_sustain: int = 25
    drop_lag: int = 15              # 信号 A 的 EMA 滞后补偿（跌落检出比真实
    #                                 切换晚 ~5-10 步，截断时往回退）
    warmup: int = 250
    boundary_cooldown: int = 120
    dream_grace: int = 20
    # 代谢
    prune_abs: float = 0.40           # 凋亡线：只清剪真正失败者（最佳领地
    #                                 可靠性 <0.40）。竞选失利者（~0.45）与
    #                                 学习中的细胞（0.5 起步）必须活下来——
    #                                 前者是自由细胞储备，后者正在爬坡。
    split_acc: float = 0.95          # 集成精度低于此才考虑分裂
    split_rel_frac: float = 0.0      # >0 时改按峰值比例判（split_ref =
    #                                 split_rel_frac × peak_acc）——规模无关，
    #                                 真实文本精度平台远低于 0.95 的绝对线
    split_cooldown: int = 300        # 分裂冷却（步）：防止切换后跌落期连环分裂
    split_noise: float = 0.3
    counterfactual: bool = True
    cf_noise: float = 0.08
    dream_enabled: bool = True
    router: str = "divergence"
    dist_refresh: int = 10          # 全库 forward 刷新熵缓存的间隔（步）
    seed: int = 0
    # ---- 规模实验（run_scale.py）专用：引擎与词表 ----
    backend: str = "numpy"          # "numpy"（玩具/复现路径）| "torch"（LLM 规模）
    vocab: int = 16                 # 词表大小（numpy 路固定 16；torch 路=BPE 词表）
    grad_clip: float = 1.0          # torch 路梯度裁剪（0=不裁剪）
    weight_decay: float = 0.0       # torch 路 AdamW weight decay
    acc_prior: float = 0.34         # 边界信号 A 的峰值先验（玩具任务标定 0.34；
    #                                 真实文本 next-token 精度从 ~1/V 爬升，
    #                                 先验高于早期精度会造成假跌落——规模实验
    #                                 传 0.0 让峰值从首步起跟随实测）
    out_full: bool = False          # True=全序列 next-token 计分（真实 LM）；
    #                                 False=只用 SEP 后输出段（玩具任务）
    gauss_chi: float = 250.0        # 签名马氏离群剔除（≤0 时按 3.9×d_model 自动缩放，
    #                                 维度越高厚尾越重，固定值只对 64 维标定过）
    amp_bf16: bool = False          # torch 路 bf16 autocast（CPU AVX512-BF16 ~1.7x；
    #                                 参数/Adam 保持 fp32，仅激活值降精度）


class OBMLM:
    def __init__(self, cfg: OBMLMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # 概念签名投影：冻结（不训练），全部细胞共享——签名空间全程一致，
        # 细胞分裂/凋亡不改变领地几何（只有 cells[0] 实际被用于签名）
        self.sig_emb = self.rng.normal(0.0, 1.0, (cfg.vocab, cfg.d_model))
        self.cells = [bk.make_cell(self.rng, cfg.vocab, cfg.d_model,
                                   cfg.n_heads, cfg.n_layers, cfg.max_len, i,
                                   sig_emb=self.sig_emb, cfg=cfg)
                      for i in range(cfg.n_cells_init)]
        self.next_id = cfg.n_cells_init
        self.step = 0
        self.energy = 0.0
        self.buffer = []             # 当前段经历 [(seq, tgt, ens_acc, step)]
        self.segments = []           # 封存段 [{id, samples, n, gmean, gvar}]
        self.seg_counter = 0
        # 签名空间运行统计（领地匹配 + 信号 B）
        self.cur_x = dict(sum=np.zeros(cfg.d_model), sq=np.zeros(cfg.d_model), n=0)
        self.seg_ref = None
        self.ew = None
        self.d_run = 0
        self.d_start = None
        self.d_last = None          # 当前签名空间分布距离（诊断用）
        self.drop_start = None
        self.boundary_step = None
        self.ema_acc = cfg.acc_prior   # 二分类任务随机水平上方的先验
        self.peak_acc = cfg.acc_prior
        self.drop_run = 0
        self.last_seal = -10 ** 9
        self.last_birth = -10 ** 9
        self.dream_grace = 0
        self.award = None            # 单份授予任期锁 {cid, seg, step, best}
        # 任务计分 mask：玩具任务只算 SEP 后输出段；真实 LM 全序列 next-token
        self.out_mask = (np.ones(cfg.max_len, dtype=bool) if cfg.out_full
                         else _OUT_MASK)
        self.events = []
        self.history = []
        self._dist_cache = None      # 全库平均 next-token 分布（anchor 用）
        self._top_cache = None       # 全库输出段 top-1 概率均值（反射弧用）
        self._probe_cache = None     # 全库输出段精度（合同授予探针用）
        self._dist_cache_step = -10 ** 9
        self._seg_peak_acc = {}      # 段内全库峰值 next-token 精度（相对记账线参考）

    # ================= 领地匹配 =================
    @staticmethod
    def _gauss_dist(m1, v1, m2, v2):
        """两个对角高斯的对称均值距离（按方差和归一）。

        同任务复现 ≈0.3（采样误差），跨任务 ≥3.8——分离度 >10x，
        比余弦稳健（余弦扛不住任务间 token 范围重叠）。
        """
        return float(0.5 * np.sum((m1 - m2) ** 2 / (v1 + v2)))

    def _context(self, sig):
        """sig: (B,D)。返回 {segment_id: weight}。

        用"到各段签名均值的马氏距离"做软匹配，而非高斯对数似然：
        不同任务的签名方差尺度不同（shift 的输出 token 与输入不同，
        12 个独立 token，方差是 copy/reverse/sort——6 个重复 token——
        的一半），LL 的 log-方差项会让所有输入都更匹配低方差段
        （实测 copy 输入 96% 误配 shift 段，领地识别完全错乱）。
        距离判据无此混淆（与段级合并同一判据，互洽）。
        """
        cfg = self.cfg
        x = sig.mean(axis=0)                          # 一步一个 batch，取均值
        cand = []
        for s in self.segments:
            if s["n"] >= cfg.ctx_min_seg_n and s.get("gmean") is not None:
                cand.append((s["id"], s["gmean"], s["gvar"]))
        cx = self.cur_x
        if cx["n"] >= cfg.ctx_min_seg_n:
            m = cx["sum"] / cx["n"]
            v = np.maximum(cx["sq"] / cx["n"] - m * m, cfg.var_floor)
            cand.append((self.seg_counter, m, v))
        if not cand:
            return {self.seg_counter: 1.0}
        ds = np.array([self._gauss_dist(x, cfg.var_floor * np.ones_like(x),
                                        m, v) for _, m, v in cand])
        w = np.exp(-(ds - ds.min()) / cfg.ctx_temp)
        w /= w.sum()
        w = (1.0 - cfg.ctx_uniform) * w + cfg.ctx_uniform / len(w)
        return {sid: float(wi) for (sid, _, _), wi in zip(cand, w)}

    def _reprise_seg(self, sig):
        """本 batch 签名是否贴近某个**已封存**段（任务复现）？返回段 id 或 None。

        只用原始马氏距离判据（与段级合并同族），不用 ctx 权重——ctx 的
        softmax 在 ctx_temp=0.3 下对异任务批次也饱和（实测 shift 批次到
        copy 段距离 5.08 仍拿 ~0.95 权重），拿权重设门限会把专才送给异
        任务。距离门限 3.0：同任务训练批次 ≈1.8±1.0（段库存样本实测），
        覆盖 ~85%；最近的跨任务对（shift→reverse ≈4.5±2.7）大部分被挡
        在门外，尾部漏网的由 _select 的精度门限（probe ≥ 0.5）拦截——
        单批距离本身分不干净这两条分布，必须两道门限叠加。
        当前段（未封存）不参与——它的统计太少，复现判定交给封存时的
        段级合并；这里的职责是在封存前的混淆窗口内认出复现。
        """
        cfg = self.cfg
        if not self.segments:
            return None
        x = sig.mean(axis=0)
        best, best_d = None, None
        for s in self.segments:
            if s.get("gmean") is None:
                continue
            d = self._gauss_dist(x, cfg.var_floor * np.ones_like(x),
                                 s["gmean"], s["gvar"])
            if best_d is None or d < best_d:
                best, best_d = s["id"], d
        return best if (best_d is not None and best_d <= cfg.award_reprise_d) else None

    # ================= 路由 =================
    def _note_bar(self, cell, seg_id):
        """记账门限：随细胞在该段的资历（seg_rel）从 fulfill_acc 升到 expert_acc。

        资历浅的细胞（正在学）用低线，允许难任务突破平台期（copy 相变）；
        资历深的细胞（已建声誉）用高标准，退化会被如实记违约，信誉随
        之下滑、自动退出合同池——防止"半吊子"细胞把 seg_rel 刷到 1.0
        后在评估集成里冒充专才（实测 copy 评估 3 个 0.98 信誉细胞实际
        精度 0.43-0.50，集成被拖垮）。
        """
        cfg = self.cfg
        r = cell.seg_rel.get(seg_id, 0.5)
        t = min(1.0, max(0.0, (r - 0.5) / 0.4))
        return cfg.fulfill_acc + (cfg.expert_acc - cfg.fulfill_acc) * t

    def _valid(self, rels, probe=None, commitments=None, force_award=False,
               training=False, reprise=None, reprise_seg=None):
        """合同池：cond_rel 过线的细胞。无人过线（或强制授予）时，把合同
        授予单个"最自由的候选人"：

        - 自由细胞（承诺度 = 各段 seg_best 峰值低于 specialist_line）优先——
          专才细胞有存量知识要保护，不让它去接异乡合同；
        - 自由细胞中用 probe（当前 batch 输出段精度）挑最有潜力的；
        - 无自由细胞时（容量耗尽）才在专才中按 probe 挑选——
          此时该任务的复兴依赖任务复现或分裂补充容量。

        MVP 的"回退训练全部细胞"在 LM 下是知识破坏的主通道（8 个细胞
        同时以违约 lr 训练所有任务），改为单份授予。

        training=True 时启用**任期锁**：授予不是每步重抽的彩票，而是
        黏着到受授细胞"毕业"（seg_rel 爬过合同线，合同池接管）、"无
        进展"（任内最佳 batch 精度始终低于 stall 线）或超时为止。切换
        窗口内受授细胞易手 = 谁都学不会当前任务（违约记账轮作），见
        cfg.award_* 注释里的实测崩塌链路。评估/梦境路径（training=False）
        不走任期锁，保持原有选路语义。

        **复现授予**（training 且 reprise 非空）：本 batch 签名贴近某个
        已封存段（任务复现，距离门限 award_reprise_d），该段有专才
        （seg_rel ≥ specialist_line）且专才在本 batch 输出段精度过线
        （probe ≥ award_reprise_probe——距离尾部偶发放进的异任务专才
        由此挡驾）时，合同直接授予该专才，任期封顶 award_reprise_tenure。
        混淆窗口里"从零学旧任务的自由细胞"此前能赢 probe 彩票（实测
        seed 2 复现在线恢复只有 0.596，而该段专才的存量精度是 0.97）
        ——复现不是新任务，没有理由不让存量知识立即上岗。置于合同池
        早退之后（稳定期仍由合同池说话）、任期锁之前（确认的复现强于
        锁的防重抽意图）。
        """
        cfg = self.cfg
        idx = [i for i in range(len(rels)) if rels[i] >= cfg.contract_floor]
        if idx and not force_award:
            if training:
                self.award = None      # 合同池已有人：授予期结束
            return idx
        # ---- 评估/梦境/生成路径（training=False）----
        # 无合同覆盖时不走授予机器（任期锁/探针彩票/复现授予都是训练期
        # 概念）：交还全库，由 _select 按 cond_rel 排序取 top-k。评估选路
        # 必须反映积累的条件可靠性——pilot 实测陈旧探针缓存（上次训练批
        # 的精度）把 code_py 输入全部授给上次受训的通才细胞，路由 ppl
        # 97861 vs 专才 1332（73×），选路即评估质量，不能交给彩票。
        if not training:
            return list(range(len(rels)))
        # ---- 复现授予：专才直接接管复现的旧任务 ----
        if training and reprise is not None:
            self.award = dict(cid=self.cells[reprise].cell_id,
                              seg=self.seg_counter, step=self.step,
                              best=0.0, match=reprise_seg,
                              cap=cfg.award_reprise_tenure)
            return [reprise]
        n = len(rels)
        if commitments is None:
            commitments = np.array([max(c.seg_best.values(), default=0.0)
                                    for c in self.cells])
        # ---- 任期锁：授予有效期内不接受重抽 ----
        if training and self.award is not None:
            a = self.award
            ids = [c.cell_id for c in self.cells]
            if a["cid"] not in ids:
                self.award = None                    # 细胞已凋亡：授予失效
            else:
                i = ids.index(a["cid"])
                cell = self.cells[i]
                # 毕业判据按**当前段**算（含复现授予）：受授专才在旧段的
                # seg_rel 本来就高（那正是它当选的原因），按旧段判它第一步
                # 就"毕业"，授予只存活一步、窗口交还自由细胞彩票；按当前段
                # 判，它要在本段重新记账建立合同才毕业——存疑期不记账，
                # 窗口关闭后 ~2 步爬过线，与合同池无缝交接。
                r = cell.seg_rel.get(self.seg_counter, 0.5)
                tenure = self.step - a["step"]
                cap = a.get("cap", cfg.award_max_tenure)
                if r >= cfg.contract_floor:
                    self.award = None                # 毕业：合同池接管
                elif tenure >= cap:
                    self.award = None                # 超时保险：释放重授
                elif (tenure >= cfg.award_min_tenure
                        and a["best"] < cfg.award_stall_acc):
                    self.award = None                # 无进展：释放重授
                else:
                    return [i]                       # 任期有效
        free = np.where(commitments < cfg.specialist_line)[0]
        cand = free if len(free) > 0 else np.arange(n)
        if training and probe is not None and len(probe) == n:
            j = int(cand[int(np.argmax(np.asarray(probe)[cand]))])
            if training:
                self.award = dict(cid=self.cells[j].cell_id,
                                  seg=self.seg_counter, step=self.step,
                                  best=0.0)
            return [j]
        if len(cand) == 1:
            j = int(cand[0])
            if training:
                self.award = dict(cid=self.cells[j].cell_id,
                                  seg=self.seg_counter, step=self.step,
                                  best=0.0)
            return [j]
        return [int(x) for x in cand]

    def _refresh_dist_cache(self, idx, mask=None, tgt=None):
        """全库刷新三样缓存：
        - 平均 next-token 分布（分歧路由的锚点熵用）
        - 输出段 top-1 概率均值（反射弧的置信度用）
        - 输出段精度（合同授予的探针用；tgt 为 None 时不刷新探针）

        torch 路走 forward_stats（torch 内分块统计）——V=16K 时把 (B,L,V)
        的 softmax 搬进 numpy 是 400MB/细胞/步，实测拖慢 4 倍。
        """
        out, tops, probes = [], [], []
        for c in self.cells:
            st = bk.forward_stats(c, idx, mask, tgt)
            out.append(st["dist"])
            tops.append(st["top1"])
            if tgt is not None:
                probes.append(st["probe"])
        self._dist_cache = out
        self._top_cache = tops
        if tgt is not None:
            self._probe_cache = probes
            # 段内全库峰值：相对记账线的参考（探针覆盖所有细胞，不受
            # 授予格局限制——通才垄断授予时峰值仍反映真实最优水平）
            pk = float(max(probes)) if probes else 0.0
            self._seg_peak_acc[self.seg_counter] = max(
                self._seg_peak_acc.get(self.seg_counter, 0.0), pk)
        self._dist_cache_step = self.step

    def _cell_dists(self, idx):
        if self._dist_cache is None or self.step - self._dist_cache_step >= self.cfg.dist_refresh:
            self._refresh_dist_cache(idx)
        return self._dist_cache

    def _select(self, rels, idx=None, mask=None, refresh=False,
                allow_reflex=True, tgt=None, probe=None, ctx=None,
                training=False, reprise_seg=None):
        cfg = self.cfg
        # 刷新判定：间隔制（dist_refresh）——decide 不再强制每步全库前向，
        # 那在 V=16K 时是 4 细胞 × ~0.6s 的固定税；refresh=True 仅评估/
        # 诊断路径强制刷新时保留。
        if idx is not None and (refresh or self._dist_cache is None
                                or self.step - self._dist_cache_step >= cfg.dist_refresh):
            self._refresh_dist_cache(idx, mask, tgt)
        if probe is None:
            probe = getattr(self, "_probe_cache", None)
        if probe is not None and len(probe) != len(self.cells):
            probe = None
        # 承诺度按峰值判定：瞬时记账污染（切换窗口的违约记账）不应把
        # 专才误判为自由细胞——专才的存量知识要靠它保护
        commitments = np.array([max(c.seg_best.values(), default=0.0)
                                for c in self.cells])
        # 合同池有效性（仅训练路径）：两个条件任不满足，就认为"当前输入
        # 不属于稳定的当前段"——切换后的混淆窗口（新段统计未建立/未封存，
        # ctx 仍指向旧段；或边界嫌疑已触发），旧专才会以高 cond_rel 过线
        # 被拉去训练新任务：实测 20 步就把 sort 专才的信誉从 1.0 砍到
        # 0.37（评估时被排除，sort 崩塌），且权重开始向新任务漂移。
        # 窗口内强制走单份授予，让自由细胞承接新任务。
        pool_valid = True
        if training:
            ctx_cur = (ctx or {}).get(self.seg_counter, 0.0)
            pool_valid = (ctx_cur >= 0.3
                          and self.d_run == 0 and self.drop_run == 0)
        # 复现授予的候选专才：该段承诺度（seg_rel，非峰值——erode 只作用于
        # 当前段 id，专才在旧段的 seg_rel 封存后不再衰减，实测维持在 0.95+）
        # 超过 specialist_line 的细胞中挑最高者。低于该线的"复兴中的前任"
        # 不参与——它自己还在恢复期，不该被再次推上前线。
        # 再加一道精度门限：专才必须在本 batch 输出段真有本事（probe ≥
        # award_reprise_probe）。距离门限的尾部偶发会把异任务专才放进来
        # （shift 批次与 reverse 段的签名距离 4.5±2.7，尾部可低至 ~2.8），
        # 而跨任务精度实测 ≤0.36、同任务 ≥0.90——0.5 的精度门限把异任务
        # 专才挡在门外，复现授予只发生在"存量知识确实能上岗"的批次上。
        reprise = None
        if training and reprise_seg is not None:
            cand = [i for i, c in enumerate(self.cells)
                    if c.seg_rel.get(reprise_seg, 0.0) >= cfg.specialist_line]
            if cand:
                j = max(cand,
                        key=lambda i: self.cells[i].seg_rel[reprise_seg])
                if (probe is not None and len(probe) == len(self.cells)
                        and probe[j] >= cfg.award_reprise_probe):
                    reprise = j
        pool = self._valid(rels, probe=probe, commitments=commitments,
                           force_award=not pool_valid, training=training,
                           reprise=reprise, reprise_seg=reprise_seg)
        pool_rels = rels[pool]
        # 非训练路径（评估/生成/梦境核算）：合同池或全库，一律按条件
        # 可靠性排序取 top-k。不走分歧路由——那是训练期的探索机制
        # （锚点 = 熵最大的细胞，通常是没学会当前任务的自由细胞），
        # 评估期把输入交给"最被证明擅长该上下文"的细胞。
        if not training:
            k = min(cfg.k_max, len(pool))
            return [pool[i] for i in np.argsort(pool_rels)[::-1][:k]], False
        best_local = int(np.argmax(pool_rels))
        best = pool[best_local]
        conf = 0.0
        if self._top_cache is not None and best < len(self._top_cache):
            conf = self._top_cache[best]
        # 反射弧：条件可靠性最高且对输出段高度自信 -> 单跳直达
        if (allow_reflex and pool_rels[best_local] >= cfg.reflex_rel
                and conf >= cfg.reflex_conf):
            return [best], True
        dists = self._dist_cache if self._dist_cache is not None else \
            [np.full(self.cfg.vocab, 1.0 / self.cfg.vocab)] * len(self.cells)
        # 缓存与细胞数对齐：凋亡后缓存可能更长（截断）；分裂后缓存可能
        # 更短（birth 不失效缓存，评估路径不刷新）——补均匀分布占位，
        # 否则 pool 索引越界（divergence_route 实测 IndexError）。
        if len(dists) > len(self.cells):
            dists = dists[:len(self.cells)]
        elif len(dists) < len(self.cells):
            dists = list(dists) + [np.full(self.cfg.vocab, 1.0 / self.cfg.vocab)] * (len(self.cells) - len(dists))
        k = min(cfg.k_max, len(pool) + 1)
        if cfg.router == "similarity":
            return [pool[i] for i in np.argsort(pool_rels)[::-1][:k]], False
        # 锚点收窄到合同池内（strict_pool）：MVP 的"全库锚点自由探索"
        # 在 LM 下会把旧专才拉去训练异任务——锚点是熵最大的细胞，通常
        # 正是没学会当前任务的自由细胞/衰退老专家，越池即破坏。LM 的
        # 复兴通道是 cond_rel 重回池 + 窗口期 probe 授予，不靠锚点曝光。
        ent = [-float(np.sum(d * np.log(d + _EPS))) for d in dists]
        anchor = int(pool[int(np.argmax([ent[i] for i in pool]))])
        return divergence_route(dists, k, anchor=anchor, pool=pool,
                                strict_pool=True), False

    # ================= 决策 =================
    def decide(self, idx, sig, tgt, mask):
        """返回 dict：ctx/rels/selected/reflex/ens_acc/cell_accs/logits(T)/weights。"""
        cfg = self.cfg
        ctx = self._context(sig)
        rels = np.array([c.cond_reliability(ctx) for c in self.cells])
        # 刷新走 dist_refresh 间隔（_select 内部判定），不强制每步全库前向
        selected, reflex = self._select(rels, idx, mask, refresh=False, tgt=tgt,
                                        ctx=ctx, training=True,
                                        reprise_seg=self._reprise_seg(sig))
        logits_T = []
        for i in selected:
            c = self.cells[i]
            logits_T.append(bk.forward(c, idx, train=True))
        w = np.exp(cfg.ensemble_beta * rels[selected])
        w = w / w.sum()
        # 融合统计：per-cell top1 + ensemble acc 一趟 V 维算完（单细胞时
        # ensemble argmax ≡ per-cell argmax，softmax 单调，省一次 fp32 物化）
        accs, ens_acc = bk.decide_stats(self.cells, selected, w,
                                        logits_T, tgt, mask)
        self.energy += sum(self.cells[i].n_params() for i in selected)
        return dict(ctx=ctx, rels=rels, selected=selected, reflex=reflex,
                    ens_acc=ens_acc, cell_accs=accs, logits_T=logits_T,
                    weights=w)

    # ================= 单步生命周期 =================
    def run_step(self, seq, tgt, mask):
        cfg = self.cfg
        # 1) 感知：签名（共享 token embedding 的均值池化）
        sig = self.cells[0].signature(seq)
        # 2) 领地惰性衰减（先于记账：note 会刷新 seg_note，erode 的 dt 才有意义）
        for c in self.cells:
            c.erode(self.seg_counter, self.step, cfg.inactivity_decay,
                    cfg.commitment_decay)
        # 2.5) 信号 B 先行更新：边界嫌疑（d_run）必须在决策前就绪——
        #     切换首步签名距离即跳变，若等决策后再更新，旧专才会在
        #     切换第一步被误选进池训练新任务（违约记账污染旧段信誉）。
        self._update_sig_stats(sig)
        # 3) 决策（含全库熵缓存刷新）
        d = self.decide(seq, sig, tgt, mask)
        selected, reflex = d["selected"], d["reflex"]
        # 3.5) 记账存疑豁免：本 batch 输出段精度相对峰值深度跌落（信号 A 的
        #      原始步版本，比 EMA 早 ~3 步发现切换）、或边界嫌疑已持续
        #      （d_run/drop_run>0）时，数据的段归属不可信——不订合同。
        #      切换首步封存尚未发生（seg_counter 仍指旧段、ctx 仍指向旧段
        #      统计），旧专才可能被选进池训练异任务；此时把异任务精度
        #      记账到旧段信誉就是知识破坏（实测 3 步把 r 从 0.9 砍到 0.66，
        #      专才掉到承诺度线以下被授予异乡合同，最终转化成异任务专才）。
        #      权重照常更新（受授予细胞必须能在封存前学到新任务），
        #      只是存疑期不订合同；封存后 peak/EMA 复位，记账立即恢复。
        #      第三重：ctx（本 batch 的段先验分布）不指向当前段——
        #      分布距离信号依赖签名可分性，copy↔shift 的 token 范围高度
        #      重叠（1-14 vs 1-9）使距离超不过 d_thr，d_run 永不点火；
        #      但 ctx 直接读先验：若当前段根本不是本 batch 的真实归属，
        #      记账就是张冠李戴（实测 copy 复现首步把 0.99 的 copy 精度
        #      记进 shift 段账，seg_rel(#3) 被刷到 0.55 堂堂毕业，该细胞
        #      的梦从此按权重混入 38% shift 样本自我污染，copy 二次崩塌）。
        suspect = (d["ens_acc"] < self.peak_acc - cfg.acc_drop
                   or self.drop_run > 0 or self.d_run > 0
                   or d["ctx"].get(self.seg_counter, 0.0) < 0.3)
        # 3.3) 段内峰值并入受授细胞的实时精度（同样受存疑豁免门控——
        #     切换窗口的异域精度不配当旧段的峰值参考）：探针缓存每
        #     dist_refresh（10）步才刷新，授予细胞在间隔内的进步不漏计。
        if not suspect and d["cell_accs"]:
            pk = float(max(d["cell_accs"]))
            if pk > self._seg_peak_acc.get(self.seg_counter, 0.0):
                self._seg_peak_acc[self.seg_counter] = pk
        # 4) 三因子学习 + 条件合同记账（note-on-current）
        #    两个门限分开：学习率门限宽松（fulfill_acc）——允许难任务突破
        #    平台期、允许退化专才全额恢复；记账门限严格（_note_bar，随资历
        #    上升）——信誉必须反映真实水平，半吊子细胞刷不上 1.0。
        for j, i in enumerate(selected):
            c = self.cells[i]
            acc_j = d["cell_accs"][j]
            lr = cfg.lr if acc_j >= cfg.fulfill_acc else cfg.lr_breach
            loss = d["logits_T"][j]
            # 用已建好的图反传（全序列 LM 损失）；引擎分派在 backend
            bk.ce_step(self, c, loss, tgt, lr)
            if not suspect:
                # 记账判定：绝对线（随资历上升，防半吊子刷信誉）或相对线
                # （≥ 段内全库峰值 × note_peak_frac，难域专才的上升通道）
                # 任一满足即记功。两条线服务同一目的——seg_rel 反映
                # "本细胞在本段的真实排位"，玩具精度尺度下绝对线主导，
                # LM 难域（最优 acc 也 < 0.12）由相对线主导。
                bar = self._note_bar(c, self.seg_counter)
                peak = self._seg_peak_acc.get(self.seg_counter, 0.0)
                ok = acc_j >= bar or (peak > 0.0 and
                                      acc_j >= cfg.note_peak_frac * peak)
                c.note(self.seg_counter, ok, cfg.reliability_alpha)
                c.seg_note[self.seg_counter] = self.step
        # 任期锁的进展跟踪：受授细胞在本 batch 的输出段精度入账
        # （无进展判据用——见 cfg.award_stall_acc）
        if self.award is not None:
            for j, i in enumerate(selected):
                if self.cells[i].cell_id == self.award["cid"]:
                    self.award["best"] = max(self.award["best"],
                                             d["cell_accs"][j])
        # 5) 信号 A：输出段精度 EMA
        self.ema_acc = 0.9 * self.ema_acc + 0.1 * d["ens_acc"]
        self.peak_acc = max(self.peak_acc * 0.9997, self.ema_acc)
        dropping = self.ema_acc < self.peak_acc - cfg.acc_drop
        if dropping and self.drop_run == 0:
            self.drop_start = self.step
        self.drop_run = self.drop_run + 1 if dropping else 0
        if not dropping:
            self.drop_start = None
        # 6) 经历缓冲
        for b in range(seq.shape[0]):
            self.buffer.append((seq[b].copy(), tgt[b].copy(), d["ens_acc"], self.step))
        if len(self.buffer) > cfg.buffer_size:
            self.buffer = self.buffer[-cfg.buffer_size:]
        # 8) 梦境代谢
        if cfg.dream_enabled and self.step > 0 and self.step % cfg.dream_period == 0:
            self.dream()
            self.dream_grace = cfg.dream_grace
        else:
            self.dream_grace = max(0, self.dream_grace - 1)
        # 9) 边界检测（双信号 OR，都要持续确认）
        if (self.dream_grace == 0
                and self.step > cfg.warmup
                and self.step - self.last_seal > cfg.boundary_cooldown
                and (self.drop_run >= cfg.drop_sustain or self.d_run >= cfg.d_sustain)):
            starts = [t for t in (self.d_start, self.drop_start) if t is not None]
            self.boundary_step = min(starts) if starts else self.step
            self._seal_segment()
            self.peak_acc = self.ema_acc
            self.drop_run = 0
            self.d_run = 0
        self.history.append(dict(step=self.step, acc=d["ens_acc"], k=len(selected),
                                 reflex=int(reflex), n_cells=len(self.cells),
                                 n_params=sum(c.n_params() for c in self.cells),
                                 energy=self.energy, d=self.d_last,
                                 ema=self.ema_acc, n_seg=len(self.segments)))
        self.step += 1
        return d

    def _opt_state(self, cell):
        if not hasattr(cell, "_opt"):
            cell._opt = {}
        return cell._opt

    # ================= 签名统计 + 信号 B =================
    def _update_sig_stats(self, sig):
        cfg = self.cfg
        cx = self.cur_x
        for x in sig:
            cx["sum"] += x
            cx["sq"] += x * x
            cx["n"] += 1
        if self.seg_ref is None:
            if cx["n"] >= cfg.ctx_freeze_n:
                n = cx["n"]
                m = cx["sum"] / n
                v = np.maximum(cx["sq"] / n - m * m, cfg.var_floor)
                self.seg_ref = (m, v)
                self.ew = (m.copy(), v.copy())
            return
        for x in sig:
            m0, v0 = self.ew
            diff = x - m0
            m1 = m0 + cfg.ew_alpha * diff
            v1 = (1.0 - cfg.ew_alpha) * (v0 + cfg.ew_alpha * diff * diff)
            self.ew = (m1, np.maximum(v1, cfg.var_floor))
            mr, vr = self.seg_ref
            d = float(np.sum((m1 - mr) ** 2 / (vr + v1)
                             + (np.log(v1) - np.log(vr)) ** 2))
        # 阈值判定按"步"计：一个 batch 只判定一次（MVP 的逐步语义；
        # batch 内任何样本低于阈值都清零太脆弱——12 条里 1 条接近就复位）
        self.d_last = d
        if d > cfg.d_thr:
            if self.d_run == 0:
                self.d_start = self.step
            self.d_run += 1
        else:
            self.d_run = 0
            self.d_start = None

    # ================= 封存 =================
    def _seg_gauss(self, entries):
        """签名空间高斯（马氏离群剔除，阈值按维度缩放）。"""
        cfg = self.cfg
        xs = np.array([self.cells[0].signature(e[0][None])[0] for e in entries])
        m = xs.mean(axis=0)
        v = np.maximum(xs.var(axis=0), cfg.var_floor)
        chi = cfg.gauss_chi if cfg.gauss_chi > 0 else 3.9 * cfg.d_model
        keep = np.sum((xs - m) ** 2 / v, axis=1) < chi
        if keep.sum() >= 20:
            xs = xs[keep]
            m = xs.mean(axis=0)
            v = np.maximum(xs.var(axis=0), cfg.var_floor)
        return m, v, len(xs)

    def _seal_segment(self):
        cfg = self.cfg
        if len(self.buffer) < 20:
            return
        entries = sorted(self.buffer, key=lambda t: t[3])
        # 截断点：双信号各自估计的真实切换步。
        # 信号 B 的 d_start 即切换首步（该步的 batch 已属新任务，须严格排除）；
        # 信号 A 的 drop_start 比切换晚 ~5-10 步（EMA 滞后），须回退补偿。
        # MVP 教训：新任务样本漏进旧段 buffer 会污染段高斯（7.5% 混入
        # 就把段间距离从 8.6 压到 1.28，直接造成误合并）。
        cuts = []
        if self.d_start is not None:
            cuts.append(self.d_start)
        if self.drop_start is not None:
            cuts.append(self.drop_start - cfg.drop_lag)
        cut_step = min(cuts) if cuts else None
        if cut_step is not None:
            entries = [t for t in entries if t[3] < cut_step]
            if len(entries) < 20:
                return
        cut = int(len(entries) * cfg.buf_trim)
        entries = entries[cut:]
        if len(entries) < 20:
            return
        gm, gv, gn = self._seg_gauss(entries)

        # 段级合并：签名高斯对称距离 => 任务复现。
        # 用高斯距离而非余弦：不同任务的 token 范围有重叠（reverse 1-7 与
        # shift 1-9 的签名余弦期望 ≈0.88，reverse 与 copy ≈0.71），余弦判据
        # 会把新段并错段（实测 reverse 段误并入 copy 段）；高斯距离下同任务
        # 复现 ≈0.3、跨任务 ≥3.8，分离干净。取距离最小者。
        best_dist, best_seg = None, None
        for s in self.segments:
            if s.get("gvar") is None:
                continue
            dist = self._gauss_dist(gm, gv, s["gmean"], s["gvar"])
            if best_dist is None or dist < best_dist:
                best_dist, best_seg = dist, s
        if best_seg is not None and best_dist <= cfg.seg_merge_dist:
            s = best_seg
            s["n"] += gn
            s["gmean"] = 0.7 * s["gmean"] + 0.3 * gm
            s["gvar"] = 0.7 * s["gvar"] + 0.3 * gv + 0.21 * (s["gmean"] - gm) ** 2
            s["samples"].extend([(e[0], e[1]) for e in entries])
            if len(s["samples"]) > cfg.seg_buffer_size:
                s["samples"] = s["samples"][-cfg.seg_buffer_size:]
            for cell in self.cells:
                v = cell.seg_rel.pop(self.seg_counter, None)
                b = cell.seg_best.pop(self.seg_counter, None)
                if v is not None:
                    old = cell.seg_rel.get(s["id"])
                    cell.seg_rel[s["id"]] = (0.7 * old + 0.3 * v) if old is not None else v
                    if b is not None:
                        # 承诺度峰值取 max：细胞确实在这片领地达到过 b
                        cell.seg_best[s["id"]] = max(cell.seg_best.get(s["id"], 0.0), b)
                    cell.seg_note.pop(self.seg_counter, None)
            self.seg_counter += 1
            self._reset_current()
            self.events.append((self.step, "seal",
                                f"任务复现：{len(entries)} 个片段并入概念段 #{s['id']}"
                                f"（距离 {best_dist:.2f}）"))
            return

        for cell in self.cells:
            cell.freeze(self.seg_counter, self.step, cfg.inactivity_decay,
                        cfg.commitment_decay)
        seg_entries = sorted(entries, key=lambda t: -t[2])[:cfg.seg_buffer_size]
        self.segments.append({"id": self.seg_counter,
                              "samples": [(e[0], e[1]) for e in seg_entries],
                              "n": gn, "gmean": gm, "gvar": gv})
        new_id = self.seg_counter
        self.seg_counter += 1
        if len(self.segments) > cfg.max_segments:
            dropped = self.segments.pop(0)
            self.events.append((self.step, "forget", f"最老概念段 #{dropped['id']} 被挤出"))
        self._reset_current()
        dist_txt = f"{best_dist:.2f}" if best_dist is not None else "—"
        self.events.append((self.step, "seal",
                            f"概念段 #{new_id} 封存：n={gn}，库存 {len(self.segments)} 段，"
                            f"最近段距离 {dist_txt}"))

    def _reset_current(self):
        self.buffer = []
        self.cur_x = dict(sum=np.zeros(self.cfg.d_model),
                          sq=np.zeros(self.cfg.d_model), n=0)
        self.seg_ref = None
        self.ew = None
        self.d_run = 0
        self.d_start = None
        self.drop_start = None
        self.boundary_step = None
        self.last_seal = self.step

    # ================= 梦境代谢 =================
    def _pools(self):
        pools = [(s["id"], s["samples"]) for s in self.segments]
        if self.buffer:
            pools.append((self.seg_counter, [(e[0], e[1]) for e in self.buffer]))
        return pools

    def dream(self):
        cfg = self.cfg
        pools = self._pools()
        if not pools:
            return
        rng = self.rng
        for _ in range(cfg.dream_batches):
            for c in self.cells:
                # 专家梦域：段池按本细胞的领地可靠性加权——专才细胞梦见
                # 自己的领地（自我巩固、抗覆盖），无领地细胞均匀探索。
                # MVP 的均匀采样在 LM 下是"抹平均"：每个细胞都梦到所有
                # 任务，专才被自己的梦缓慢稀释（实测无一个细胞保持旧任务）。
                w_pool = np.array([c.seg_rel.get(sid, 0.0) for sid, _ in pools])
                if w_pool.sum() > 1e-9:
                    w_pool = w_pool / w_pool.sum()
                else:
                    w_pool = None
                seqs, tgts, seg_ids = [], [], []
                for _ in range(cfg.dream_batch):
                    sid, samples = pools[rng.choice(len(pools), p=w_pool)]
                    s, t = samples[rng.integers(len(samples))]
                    seqs.append(s); tgts.append(t); seg_ids.append(sid)
                seq = np.array(seqs); tgt = np.array(tgts)
                lg = bk.forward(c, seq, train=True,
                                noise=(cfg.cf_noise if cfg.counterfactual else 0.0)
                                if rng.random() < 0.5 else 0.0)
                bk.ce_step(self, c, lg, tgt, cfg.dream_lr)
                # 合同核对：梦里样本自带段号，按段分组核对旧账；只核对
                # 本细胞**已持有合同**的段（seg_rel ≥ 合同线）——梦境是
                # 自我巩固，不是资格考试：无合同细胞在梦里到处试错会被
                # 失败记账拖死（实测自由细胞首次梦境即凋亡，储备清空）。
                # 用输出段精度（SEP 后 6 位）而非全序列——输入段是恒等
                # 复制任务，任何细胞都能"学会"，会把履约记账灌水。
                pred = bk.argmax_pred(c, seq)
                for sid in set(seg_ids):
                    if c.seg_rel.get(sid, 0.0) < cfg.contract_floor:
                        continue
                    idx = [k for k in range(len(seg_ids)) if seg_ids[k] == sid]
                    acc = float(np.mean(pred[idx][:, self.out_mask] == tgt[idx][:, self.out_mask]))
                    c.note(sid, acc >= self._note_bar(c, sid), cfg.dream_alpha)
        # 代谢：凋亡（有失败记录的细胞；其段样本池仍留在段库存里，
        # 复兴时由任意自由细胞承接） -> 分裂
        eval_samples = []
        for _, samples in pools:
            eval_samples.extend(samples[:12])
        eval_samples = eval_samples[:96]
        seq = np.array([s for s, _ in eval_samples])
        tgt = np.array([t for _, t in eval_samples])
        ens_acc = self._ensemble_acc(seq, tgt)
        for idx in reversed(range(len(self.cells))):
            if len(self.cells) <= 2:
                break
            c = self.cells[idx]
            # 只清剪"从未建立过强专才、且已失败"的细胞：
            # 1) seg_rel 为空 = 未证明而非不可靠——自由细胞储备，
            #    新任务的合同要授予它们（单份授予制下尤其如此）；
            # 2) seg_best 峰值 ≥ specialist_line = 曾经是专才——其知识
            #    可能完好，只是账本被污染/老化，任务是复兴而非回收
            #    （实测误剪会把唯一的旧任务知识清空，复现段无人可复兴）；
            # 3) 最佳段正是当前段 = 在学细胞——难任务突破平台期前 r 会
            #    暂时低于凋亡线，此时剪枝等于每次都在培养中途放弃。
            if (not c.seg_rel or c.niche() >= cfg.prune_abs
                    or max(c.seg_best.values(), default=0.0) >= cfg.specialist_line
                    or max(c.seg_rel, key=c.seg_rel.get) == self.seg_counter):
                continue
            dead = self.cells.pop(idx)
            self.events.append((self.step, "death",
                                f"细胞 #{dead.cell_id} 凋亡（最佳领地可靠性 "
                                f"{dead.niche():.2f}），剩余 {len(self.cells)} 个"))
        # 分裂条件：集成精度低 + 不在切换后的恢复期 + 分裂冷却 + 未满编。
        # 恢复期判定（三条互补）：
        # 1) 近 200 步均值相对峰值腰斩——窗口要覆盖"旧任务高精度 + 新任务
        #    爬坡"的整个切换期（50 步窗口只含爬坡段时会把上升误判为平台）；
        # 2) 当前步精度远低于 EMA（切换当步 EMA 还没反应过来）；
        # 3) 近 50 步均值比此前 150 步高 0.05 以上——仍在爬坡的任务不是
        #    "卡住"，补容量要等平台期（实测 sort 在 0.87 爬坡中被误分裂）。
        # 真平台（稳定在 0.85）mean≈max 且 tail≈head → 不拦分裂，补容量。
        recent = [h_["acc"] for h_ in self.history[-200:]]
        in_dip = bool(recent) and float(np.mean(recent)) < 0.7 * float(np.max(recent))
        if self.history:
            cur = self.history[-1]
            in_dip = in_dip or cur["acc"] < cur["ema"] - 0.15
        if len(recent) >= 100:
            in_dip = in_dip or (float(np.mean(recent[-50:]))
                                > float(np.mean(recent[:-50])) + 0.05)
        # 分裂门限：split_rel_frac>0 时按峰值比例判（规模无关——真实文本的
        # next-token 精度平台 ~0.35，绝对线 0.95 会每秒都触发分裂；按峰值的
        # 85% 判，只有"稳定卡在峰值下方"的真平台才补容量）
        split_ref = (cfg.split_acc if cfg.split_rel_frac <= 0
                     else cfg.split_rel_frac * self.peak_acc)
        if os.environ.get("OBM_DEBUG_SPLIT"):
            import sys as _sys
            print(f"[split-dbg] step={self.step} ens_acc={ens_acc:.3f} "
                  f"peak_acc={self.peak_acc:.3f} split_ref={split_ref:.3f} "
                  f"below={'Y' if ens_acc<split_ref else 'N'} "
                  f"in_dip={'Y' if in_dip else 'N'} "
                  f"cooldown_ok={'Y' if self.step-self.last_birth>cfg.split_cooldown else 'N'} "
                  f"cells={len(self.cells)}/{cfg.max_cells}", file=_sys.stderr, flush=True)
        if (ens_acc < split_ref and not in_dip
                and self.step - self.last_birth > cfg.split_cooldown
                and len(self.cells) < cfg.max_cells):
            # 分裂目标 = 当前段（瓶颈任务）上损失最低的细胞：给卡住的任务
            # 补容量，而不是复制当前的失败者（失败者的复印件还是失败者；
            # 混合池上的最差细胞通常是"异任务专才"，复制它无意义）。
            # 子细胞继承父代的段合同与承诺度峰值（细胞分裂 = 身份传承）：
            # LM 的硬合同池下，空合同的 fresh child 条件可靠性只有 0.5
            # 先验、永远进不了池——分裂出的容量只能闲置（实测空转）。
            # 子细胞的信誉随后按自身表现独立记账，名不副实会自动退池。
            cur_samples = pools[-1][1] if pools else []
            cs = cur_samples[-48:]
            if not cs:
                cs = cur_samples[:48]
            cseq = np.array([s for s, _ in cs])
            ctgt = np.array([t for _, t in cs])
            best_i, best_l = 0, None
            for i, c in enumerate(self.cells):
                l = bk.mean_nll(c, cseq, ctgt)
                if best_l is None or l < best_l:
                    best_i, best_l = i, l
            parent = self.cells[best_i]
            child = parent.spawn(rng, cfg.split_noise, self.next_id, self.step)
            child.seg_rel = dict(parent.seg_rel)      # 身份传承：合同随分裂
            child.seg_best = dict(parent.seg_best)
            self.next_id += 1
            self.last_birth = self.step
            self.cells.append(child)
            self.events.append((self.step, "birth",
                                f"细胞 #{child.cell_id} 由 #{parent.cell_id} "
                                f"分裂诞生（当前段重放损失 {best_l:.2f}，继承段合同）"))

    def _ensemble_acc(self, seq, tgt):
        """梦境代谢的集成重放精度（输出段指标）。

        必须用 _OUT_MASK 只算 SEP 之后：输入段的 next-token 目标是随机
        抽取的下一个输入 token（7/8 个位置不可预测，仅 SEP 位可学），
        全序列计分的上限 ≈ (8×0.2+6×任务精度)/14 ≈ 0.55，永远低于
        split_line=0.95——实测每 200 步误触发一次分裂（churn），
        且集成精度的梦际对比全部失真。
        """
        ok, tot = 0, 0
        for b in range(0, len(seq), self.cfg.batch):
            chunk = seq[b:b + self.cfg.batch]
            t = tgt[b:b + self.cfg.batch]
            # ctx 按 chunk 自身的签名计算（样本池混合多段，不能用当前段
            # 一刀切）——否则 copy 样本会被 reverse 专才的预测评分，
            # ens_acc 恒低于 split 线，每 200 步误触发分裂（实测 churn）。
            ctx = self._context(self.cells[0].signature(chunk))
            rels = np.array([c.cond_reliability(ctx) for c in self.cells])
            sel, _ = self._select(rels, allow_reflex=False)
            w = np.exp(self.cfg.ensemble_beta * rels[sel])
            w = w / w.sum()
            n = int(self.out_mask.sum()) * len(chunk)
            ok += bk.ens_acc(self.cells, sel, w, chunk, t, self.out_mask) * n
            tot += n
        return ok / max(tot, 1)

    # ================= 评估 =================
    def _eval_batch(self, chunk, t, mask):
        # 评估不走反射弧：选路质量即精度，且避免用训练步的陈旧置信度
        ctx = self._context(self.cells[0].signature(chunk))
        rels = np.array([c.cond_reliability(ctx) for c in self.cells])
        # 无合同持有者时（专才凋亡等异常态），用当前 chunk 的探针精度
        # 选单细胞兜底，而非把不可靠细胞平均进集成
        probe = None
        if not any(r >= self.cfg.contract_floor for r in rels):
            probe = []
            for c in self.cells:
                pred = np.argmax(bk.forward(c, chunk), -1)
                probe.append(float(np.mean(pred[:, mask] == t[:, mask])))
        sel, _ = self._select(rels, allow_reflex=False, probe=probe)
        ps = []
        for i in sel:
            lg = bk.forward(self.cells[i], chunk)
            p = np.exp(lg - lg.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            ps.append(p)
        w = np.exp(self.cfg.ensemble_beta * rels[sel])
        w = w / w.sum()
        ens = sum(wi * p for wi, p in zip(w, ps))
        return np.argmax(ens, -1), ctx

    def evaluate(self, test_sets, masks):
        """每任务输出段 next-token 精度（纯推理，不计能耗、不改状态）。"""
        accs = []
        for (seqs, tgts), mask in zip(test_sets, masks):
            ok, tot = 0, 0
            for b in range(0, len(seqs), self.cfg.batch):
                chunk = seqs[b:b + self.cfg.batch]
                t = tgts[b:b + self.cfg.batch]
                pred, _ = self._eval_batch(chunk, t, mask)
                ok += int(np.sum(pred[:, mask] == t[:, mask]))
                tot += int(mask.sum()) * len(chunk)
            accs.append(ok / max(tot, 1))
        return accs

    def eval_detail(self, test_sets, masks):
        """诊断：每任务的 ctx top-1、各细胞在该任务上的输出段精度、
        条件可靠性、评估期实际选中的细胞与集成权重。
        用于定位"训练 1.000 / 评估崩塌"的不一致来源。"""
        rows = []
        for (seqs, tgts), mask in zip(test_sets, masks):
            chunk, t = seqs[:self.cfg.batch], tgts[:self.cfg.batch]
            ctx = self._context(self.cells[0].signature(chunk))
            rels = np.array([c.cond_reliability(ctx) for c in self.cells])
            sel, _ = self._select(rels, allow_reflex=False)
            cell_accs = []
            for c in self.cells:
                lg = bk.forward(c, chunk)
                pred = np.argmax(lg, -1)
                cell_accs.append(float(np.mean(pred[:, mask] == t[:, mask])))
            w = np.exp(self.cfg.ensemble_beta * rels[sel])
            w = w / w.sum()
            rows.append(dict(ctx_top=max(ctx, key=ctx.get),
                            ctx_w={k: round(v, 3) for k, v in
                                   sorted(ctx.items(), key=lambda kv: -kv[1])[:3]},
                            rels=[round(float(r), 3) for r in rels],
                            cell_accs=[round(a, 3) for a in cell_accs],
                            sel=[int(s) for s in sel],
                            w=[round(float(x), 3) for x in w]))
        return rows

    def territory_report(self, test_sets, seg_of_task):
        """每任务的 ctx top-1 段分布（领地识别质量诊断）。"""
        rows = []
        for ti, (seqs, tgts) in enumerate(test_sets):
            counts = {}
            for b in range(0, len(seqs), self.cfg.batch):
                chunk = seqs[b:b + self.cfg.batch]
                ctx = self._context(self.cells[0].signature(chunk))
                top = max(ctx.items(), key=lambda kv: kv[1])[0]
                counts[top] = counts.get(top, 0) + len(chunk)
            rows.append((seg_of_task[ti], counts))
        return rows

    @property
    def n_params(self):
        return sum(c.n_params() for c in self.cells)

    # ================= 规模实验：困惑度评估 =================
    def eval_ppl(self, seq, tgt, mask):
        """固定窗口的集成困惑度（路由与 _eval_batch 一致：cond_reliability
        加权 + 合同池选择）。真实 LM 的主指标（精度被词表大小钝化）。

        按 4 条序列分批：V=16K 时 (B,L,V) 全量 softmax 是 ~200MB/批，
        分批把峰值内存压到 ~70MB（评估每 100 步跑 4 域，不值得省这点
        计算但值得省内存）。
        """
        B, L = seq.shape
        nll_sum, n_tok, ok = 0.0, 0, 0
        for b0 in range(0, B, 4):
            s_, t_ = seq[b0:b0 + 4], tgt[b0:b0 + 4]
            ctxd = self._context(self.cells[0].signature(s_))
            rels = np.array([c.cond_reliability(ctxd) for c in self.cells])
            sel, _ = self._select(rels, allow_reflex=False)
            w = np.exp(self.cfg.ensemble_beta * rels[sel])
            w = w / w.sum()
            # 引擎分派：torch 路全程 torch 内算（ens 分布不落地 numpy）
            nll, n, c = bk.ens_nll(self.cells, sel, w, s_, t_)
            nll_sum += nll
            n_tok += n
            ok += c
        return float(np.exp(nll_sum / max(n_tok, 1))), ok / max(n_tok, 1)

    # ================= checkpoint =================
    def checkpoint(self):
        """完整可序列化状态（恢复后轨迹与保存点逐位一致，含 rng 与 Adam 矩）。

        可变状态一律深拷贝：cur_x 的数组是 += 就地更新的、events/history/
        buffer/segments 是 append 式增长的——浅引用会让"保存后继续训练"
        污染快照（实测恢复模型读到 mutate 后的 cur_x，选路分歧）。
        """
        import copy
        return dict(
            cfg=self.cfg, step=self.step, seg_counter=self.seg_counter,
            segments=copy.deepcopy(self.segments),
            buffer=copy.deepcopy(self.buffer), cur_x=copy.deepcopy(self.cur_x),
            seg_ref=copy.deepcopy(self.seg_ref),
            ew=copy.deepcopy(self.ew), d_run=self.d_run, d_start=self.d_start,
            d_last=self.d_last, drop_start=self.drop_start,
            boundary_step=self.boundary_step, ema_acc=self.ema_acc,
            peak_acc=self.peak_acc, drop_run=self.drop_run,
            last_seal=self.last_seal, last_birth=self.last_birth,
            dream_grace=self.dream_grace, award=copy.deepcopy(self.award),
            events=copy.deepcopy(self.events), history=copy.deepcopy(self.history),
            next_id=self.next_id, energy=self.energy,
            sig_emb=self.sig_emb,
            cells=[bk.cell_state(c) for c in self.cells],
            # 路由统计缓存必须入库：它们是**上一个 batch** 的统计，原模型
            # 下一步沿用（未到 dist_refresh 间隔），恢复模型若清零会强制
            # 刷新 -> 选路分歧 -> 轨迹漂移（实测 ens_acc 0.347 vs 0.306）
            dist_cache=copy.deepcopy(self._dist_cache),
            top_cache=copy.deepcopy(self._top_cache),
            probe_cache=copy.deepcopy(getattr(self, "_probe_cache", None)),
            dist_cache_step=self._dist_cache_step,
            seg_peak_acc=copy.deepcopy(getattr(self, "_seg_peak_acc", {})),
            rng_state=self.rng.bit_generator.state,
        )

    def restore(self, ck):
        """从 checkpoint 恢复（深拷贝：同一 ck 可多次恢复到不同 mesh，
        且恢复后的训练不会与 ck 或其他恢复体互相污染）。"""
        import copy
        self.step = ck["step"]
        self.seg_counter = ck["seg_counter"]
        self.segments = copy.deepcopy(ck["segments"])
        self.buffer = copy.deepcopy(ck["buffer"])
        self.cur_x = copy.deepcopy(ck["cur_x"])
        self.seg_ref, self.ew = copy.deepcopy(ck["seg_ref"]), copy.deepcopy(ck["ew"])
        self.d_run, self.d_start = ck["d_run"], ck["d_start"]
        self.d_last, self.drop_start = ck["d_last"], ck["drop_start"]
        self.boundary_step = ck["boundary_step"]
        self.ema_acc, self.peak_acc = ck["ema_acc"], ck["peak_acc"]
        self.drop_run = ck["drop_run"]
        self.last_seal, self.last_birth = ck["last_seal"], ck["last_birth"]
        self.dream_grace, self.award = ck["dream_grace"], copy.deepcopy(ck["award"])
        self.events, self.history = copy.deepcopy(ck["events"]), copy.deepcopy(ck["history"])
        self.next_id, self.energy = ck["next_id"], ck["energy"]
        # 冻结签名投影：用保存点的那一份（细胞与其共享引用）
        self.sig_emb = ck["sig_emb"]
        # 细胞列表按 checkpoint 重建（数量可能因分裂/凋亡与初始不同），
        # 再逐细胞载入参数 + Adam 矩 + 合同账本
        cfg = self.cfg
        self.cells = []
        for st in ck["cells"]:
            c = bk.make_cell(self.rng, cfg.vocab, cfg.d_model, cfg.n_heads,
                             cfg.n_layers, cfg.max_len, st["cell_id"],
                             self.sig_emb, cfg)
            bk.cell_load(c, st)
            self.cells.append(c)
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = ck["rng_state"]
        # 细胞的 rng 是 mesh rng 的共享引用（梦境反事实噪声用）——restore
        # 重建了 mesh.rng 对象，必须把新引用 Propagate 给细胞，否则恢复体
        # 的梦境噪声会走另一套随机流
        for c in self.cells:
            c.rng = self.rng
        # 路由统计缓存（保持"下一步沿用上批统计"的时序；老 checkpoint 没有
        # 这些键时退化为 None -> 下一步刷新，与新建 mesh 的首步行为一致）
        self._dist_cache = copy.deepcopy(ck.get("dist_cache"))
        self._top_cache = copy.deepcopy(ck.get("top_cache"))
        self._probe_cache = copy.deepcopy(ck.get("probe_cache"))
        self._dist_cache_step = ck.get("dist_cache_step", -10 ** 9)
        # 段内峰值精度（相对记账线的参考——影响 note 判定即轨迹）
        self._seg_peak_acc = copy.deepcopy(ck.get("seg_peak_acc", {}))

    # ---------- 权重-only 存档（沙箱推理友好） ----------
    def save_weights_only(self, path, dtype=np.float32):
        """只存参数 + 细胞身份 + 合同账本（不含 Adam 矩、不含大 buffer）。

        体积 ≈ 参数量 × dtype 字节；dtype=np.float16 时文件减半，便于 scnet
        远端训练后把大模型拉回 8GB 沙箱（半精度文件 + 加载端强转 fp32）。
        用于远端（scnet）训练完成后把模型拉回本地沙箱做推理 / 收尾。
        """
        import pickle
        cells = []
        for c in self.cells:
            st = c.state()
            params = {}
            for k, v in st["params"].items():
                if hasattr(v, "detach"):          # torch.Tensor
                    arr = v.detach().cpu().numpy()
                else:                              # 已是 numpy
                    arr = np.asarray(v)
                params[k] = arr.astype(dtype)
            cells.append(dict(
                cell_id=c.cell_id,
                params=params,
                seg_rel=dict(st["seg_rel"]), seg_note=dict(st["seg_note"]),
                seg_best=dict(st["seg_best"]), birth_step=st["birth_step"],
            ))
        out = dict(
            cfg=self.cfg, step=self.step, seg_counter=self.seg_counter,
            events=list(self.events), sig_emb=self.sig_emb,
            cells=cells, kind="weights_only",
            dtype="fp16" if dtype == np.float16 else "fp32",
        )
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=4)

    def load_weights_only(self, path, param_budget_bytes=None):
        """从权重-only 存档重建细胞（仅参数 + 合同账本，无 Adam 矩）。

        推理前调用：OBMLM(cfg) 建壳后，本方法替换 self.cells 即可。
        返回存档 dict（含 cfg / step / events 等元数据）。

        param_budget_bytes: 沙箱内存预算（默认 None=全载）。若非 None 且总
        参数量超限，按 sum(seg_rel) 保留最重要的细胞，保证 8GB 装得下。
        """
        import pickle, torch
        with open(path, "rb") as f:
            d = pickle.load(f)
        cfg = d["cfg"]
        self.cfg = cfg
        self.sig_emb = d["sig_emb"]
        self.step = d.get("step", 0)
        self.seg_counter = d.get("seg_counter", 0)
        self.events = list(d.get("events", []))
        cells_in = list(d["cells"])
        if param_budget_bytes:
            # 估算每细胞参数量（取其参数 shape 之和），按信誉降序裁剪
            def _nparam(cd):
                return sum(int(np.prod(v.shape)) for v in cd["params"].values())
            cells_in = sorted(cells_in,
                              key=lambda cd: float(sum(cd.get("seg_rel", {}).values()) or 0.0),
                              reverse=True)
            kept, used = [], 0
            for cd in cells_in:
                p = _nparam(cd) * 4  # 加载后 fp32
                if kept and used + p > param_budget_bytes:
                    break
                kept.append(cd); used += p
            cells_in = kept
            print(f"  [load] budget={param_budget_bytes/1e9:.1f}GB kept={len(cells_in)} "
                  f"cells params~{used/1e9:.2f}GB", flush=True)
        self.cells = []
        for cd in cells_in:
            c = bk.make_cell(self.rng, cfg.vocab, cfg.d_model, cfg.n_heads,
                             cfg.n_layers, cfg.max_len, cd["cell_id"],
                             self.sig_emb, cfg)
            for name, arr in cd["params"].items():
                t = np.asarray(arr)
                if t.dtype != np.float32:
                    t = t.astype(np.float32)
                getattr(c, name).data.copy_(torch.from_numpy(t))
            c.seg_rel = dict(cd.get("seg_rel", {}))
            c.seg_note = dict(cd.get("seg_note", {}))
            c.seg_best = dict(cd.get("seg_best", {}))
            c.birth_step = cd.get("birth_step", 0)
            c._opt = None
            self.cells.append(c)
        self.next_id = max((c.cell_id for c in self.cells), default=-1) + 1
        return d
