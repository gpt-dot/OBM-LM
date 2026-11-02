"""OuroborosMesh —— OBM 主类：清醒 / 梦境 / 代谢 的完整生命周期。

MVP v4 关键升级（领地识别从"意见指纹"改为"输入分布模型"）：
  1. 领地识别 = 输入空间分布匹配。v3 曾用意见指纹（各细胞在各类样本上的
     典型意见剖面）推断地盘，但**梦境反刍把所有细胞都练成了全任务通才**——
     实测所有细胞对所有任务的意见锐度都是 0.98~1.00，剖面行全是 one-hot，
     原始余弦恒等于 1.0，指纹没有任何区分度（机制性饿死）。
     而任务在**输入空间**的边际分布差异巨大：linear 各维方差≈1 各向同性，
     wave≈0.36，xor 前两维≈3 其余≈0.16，blobs3 多峰≈10，moons 前两维≈3.2
     其余≈0.12。感官层 tanh(S·x) 虽有非线性，但 atanh 后恰好等于 S·x，
     而 S 列满秩 —— x = (SᵀS)⁻¹Sᵀ·atanh(p) 可精确解码（误差 ~1e-14）。
     于是每段封存时保存该段经历在解码空间上的**对角高斯**（逐维均值/方差），
     推断领地 = 各段模型对当前输入的对数似然，softmax 成上下文权重。
  2. 记账只记在当前段。醒着时只经历"当下"这一个领地，功过就记在当下；
     旧领地的账只在梦境反刍时核对。上下文推断因此只影响"读取哪些合同
     来投票"，而不再污染账本——专才在别人任务上不再被误记违约。
  3. 领地惰性衰减。合同值只在"领地活跃而细胞久不履约"时随时间贬值；
     段一封存，其余细胞在该段的合同即冻结——沉睡的专才跨段保留信誉，
     任务复现时直接复活（配合分歧锚点保证曝光）。
  4. 梦境用元数据记账。反刍的样本自带段号，无需推断，直接核对旧账。
  5. 概念边界 = 分布切换。除了精度跌落（信号 A），新增分布距离（信号 B）：
     当前段近期样本的解码统计量（指数加权均值/方差）与段内冻结参照模型的
     差异——逐维均值偏移（方差归一）+ 逐维对数方差偏移。任务切换的落差
     13~90，稳态噪声 <0.7；对"精度还没掉但输入分布已经换了"的前几十步
     比信号 A 更灵敏，且不受做梦扰动影响（做梦不改变输入分布）。
     （v3 的脉冲范数是 1 维标量：段内窗口波动即达 8%、与任务间差异同量级，
     wave 段内误判 3 次；v4 初版用单样本似然跌落，对"窄任务跟在宽任务后"
     不降反升——紧凑分布声称中心，linear→xor 全线漏判，均已退役。）

一次 step 的生命周期（对应设计文档 §3.4 架构图）：
    感知 perceive → 路由 decide（反射弧 or 分歧深思）→ 响应
    → 局部学习 + 条件合同记账 observe → （周期）做梦 dream
"""
from dataclasses import dataclass
import numpy as np

from .cell import MicroCell
from .router import divergence_route, similarity_route

_EPS = 1e-12


@dataclass
class OBMConfig:
    d_in: int = 8
    n_classes: int = 3
    n_cells_init: int = 3
    n_hidden: int = 16
    max_cells: int = 8
    k_max: int = 4
    lr: float = 0.08                # 履约细胞的学习率
    lr_breach: float = 0.012        # 违约细胞的学习率（约 1/7，缓慢纠正不破坏）
    dream_lr: float = 0.05
    dream_batches: int = 8
    dream_batch: int = 32
    dream_period: int = 200
    buffer_size: int = 400
    seg_buffer_size: int = 120      # 封存段保留的高意外度片段数（按类别分层）
    seg_merge_size: int = 400       # 合并段的最大库存
    max_segments: int = 12
    reliability_alpha: float = 0.10
    dream_alpha: float = 0.05
    inactivity_decay: float = 0.9995   # 领地惰性衰减：活跃领地上久不履约则衰
    contract_floor: float = 0.55       # 合同门槛：条件可靠性低于此值的细胞不投票
    ensemble_beta: float = 10.0
    reflex_conf: float = 0.93
    reflex_rel: float = 0.75
    ctx_temp: float = 0.3              # 上下文 softmax 温度（单位：对数似然 nats）。
                                         # 同任务两段的 LL 差 ~0.3，跨任务差
                                         # 1~15 nats。1.0 时均匀底（0.05×段数）
                                         # 与弱 LL 项瓜分权重，实测 wave 输入的
                                         # ctx[#2] 仅 0.36（39% 的输入 <0.3），
                                         # 弱细胞靠别段可靠性混进选中集（wave
                                         # 集成 0.75 < 最强细胞 0.875）。0.3 把
                                         # 跨任务差放大 3.3 倍，唯一边际泄漏是
                                         # xor↔moons（0.15 nats，分布本就相似，
                                         # 且 moons 任务全员可靠故无害）
    ctx_uniform: float = 0.05          # 上下文权重的均匀底（防过度自信/新段冷启动）
    var_floor: float = 0.05            # 解码空间方差下限（防某维无波动时炸裂）
    gauss_chi: float = 40.0            # 封存高斯拟合的马氏距离离群阈值（χ²_8 99.9%≈26）
    ctx_min_seg_n: int = 24            # 少于此样本数的段不参与领地候选
                                         # （早期模型太噪，且让位给信号 A 检测边界）
    ctx_freeze_n: int = 120            # 当前段攒够这么多样本后冻结参照模型
    ew_alpha: float = 0.025            # 分布距离检测器的指数加权系数（窗宽~40步）
    d_thr: float = 2.0                 # 信号 B：分布距离阈值（稳态噪声 max~0.7，
                                         # 真实边界 13~90；xor->moons 仅 0.5 诚实
                                         # 漏检——两任务分布本就相似）
    d_sustain: int = 15                # 信号 B：超限需持续的步数
    seg_merge_thr: float = 0.80        # 段级合并：类别均值余弦阈值
    seg_merge_norm_tol: float = 1.6    # 段级合并：类别均值模长比容差
    buf_trim: float = 0.35             # 封存时丢弃缓冲最旧的比例（边界滞后污染）
    acc_drop: float = 0.20             # 概念边界A：在线精度 EMA 相对峰值跌落
    drop_sustain: int = 25             # 概念边界A：跌落需持续的步数（滤抖动）
    warmup: int = 250                  # 前多少步不检测边界（冷启动保护）
    boundary_cooldown: int = 120
    dream_grace: int = 20              # 做梦后短期内不检测边界（新生儿扰动）
    prune_abs: float = 0.55            # 凋亡：最佳领地可靠性仍接近随机 => 无用细胞
    split_acc: float = 0.88            # 蒸馏：集成在重放池上仍不达标则分裂
    split_noise: float = 0.3
    counterfactual: bool = True
    cf_noise: float = 0.08             # 反事实扰动：脉冲范数≈3，30 维各加 0.08
                                         # -> 噪声范数≈0.44（~15% 扰动），考鲁棒性；
                                         # 0.25（45% 扰动）会把所有细胞砸到 ~0.63，
                                         # 合同被噪声淹没，专才反被凋亡（实测教训）
    merge_enabled: bool = False        # 细胞级合并预留（MVP 关闭）
    dream_enabled: bool = True
    router: str = "divergence"         # or "similarity"
    seed: int = 0


class OuroborosMesh:
    def __init__(self, cfg: OBMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # 感官层：3 个冻结的随机投影（即插即用传感器，TRIZ 原理 24 中介）
        self.sensory = [self.rng.normal(0.0, 1.0 / np.sqrt(cfg.d_in), (cfg.d_in, 10))
                        for _ in range(3)]
        # 脉冲 -> 输入空间的解码器：p = tanh(P·x)（P 为 30×8 堆叠投影），
        # atanh 后 P·x 精确可逆，左伪逆 (PᵀP)⁻¹Pᵀ 还原 x（实测误差 ~1e-14）。
        # 领地分布模型工作在解码空间——任务差异（尺度/各向异性/多峰）
        # 在输入空间才可读，随机投影会把各向异性抹平。
        P = np.vstack([W.T for W in self.sensory])          # 30×8
        self.S_pinv = np.linalg.inv(P.T @ P) @ P.T          # 8×30
        self.pulse_dim = 30
        self.cells = [MicroCell(self.rng, self.pulse_dim, cfg.n_hidden,
                                cfg.n_classes, i) for i in range(cfg.n_cells_init)]
        self.next_id = cfg.n_cells_init
        self.step = 0
        self.energy = 0.0                # 累计"参数激活量"（仅统计决策）
        self.buffer = []                 # 当前概念段经历 [(p, y, surprise, step)]
        self.segments = []               # 概念段 [{"id","samples","means","sigma","n",
                                          #   "gmean","gvar"}]（g*: 解码空间高斯）
        self.seg_counter = 0             # 下一段的 id（单调递增，淘汰不换号）
        # 当前段的运行统计（领地匹配 + 封存时的最终统计）
        self.cur = dict(sums={}, ns={}, sq=np.zeros(self.pulse_dim), n=0)
        # 当前段在解码空间的运行累加器（分布模型的在线版）
        self.cur_x = dict(sum=np.zeros(cfg.d_in), sq=np.zeros(cfg.d_in), n=0)
        # 信号 B（分布切换）：当前段的冻结参照模型 + 指数加权分布距离
        self.seg_ref = None              # (mean, var) 或 None（冻结的参照）
        self.ew = None                   # (ew_mean, ew_var) 指数加权运行统计
        self.d_run = 0                   # 分布距离超限已持续的步数
        self.d_start = None              # 本轮分布距离首次超限的步
        self.drop_start = None           # 本轮精度连续跌落的起点
        self.boundary_step = None        # 本段边界触发步（封存时据此截断尾部污染）
        self.ema_acc = 0.5               # 在线精度 EMA（边界检测用）
        self.peak_acc = 0.5
        self.drop_run = 0                # 性能跌落已持续的步数
        self.last_seal = -10 ** 9
        self.dream_grace = 0             # 做梦后的边界检测宽限期
        self.events = []                 # (step, type, detail)
        self.history = []                # 逐步指标

    # ---------------- ① 感知 ----------------
    def perceive(self, x):
        """输入 -> 概念脉冲（感官细胞即插拔用，冻结不学）。"""
        return np.concatenate([np.tanh(x @ W) for W in self.sensory])

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def _decode(self, p):
        """脉冲 -> 输入空间（感官层的逆）：atanh 逐元素 + 左伪逆。

        p = tanh(P·x)  =>  atanh(p) = P·x  =>  x = (PᵀP)⁻¹Pᵀ·atanh(p)。
        """
        u = np.arctanh(np.clip(p, -1.0 + 1e-9, 1.0 - 1e-9))
        return self.S_pinv @ u

    @staticmethod
    def _diag_ll(x, m, v):
        """各维独立高斯的对数似然（去掉与段无关的常数不影响比较）。"""
        return float(-0.5 * np.sum(np.log(v) + (x - m) ** 2 / v))

    # ---------------- 领地匹配（条件合同的开关） ----------------
    def _context(self, p):
        """推断输入属于哪个概念段的地盘（输入空间分布匹配）。

        每段封存时保存该段经历在**解码空间**上的对角高斯（gmean/gvar），
        当前段用运行累加器（样本少时噪声大，设最小票数）。候选段的模型对
        当前解码输入算对数似然，softmax（温度 ctx_temp，单位 nats）成上下
        文权重，再加均匀底防过度自信。

        为什么似然（含 log 方差项）而不是别的匹配：任务间边际分布的差异
        同时体现在**尺度**（wave 0.36 vs linear 1）和**形状**（xor/moons
        前两维大其余小）上；去掉 log 方差项则宽分布永远赢（一切输入的能量
        都"典型"），保留它才是正确的模型比较。已知局限：窄分布会以正确
        的贝叶斯理由声称低范数区域（如 ~18% 的 linear 输入的范数落在
         wave 模型的典型范围内）——这是后验的真实答案，靠合同分层消化。
        """
        cfg = self.cfg
        x = self._decode(p)
        cand = []   # [(seg_id, mean, var)]
        for s in self.segments:
            if s["gvar"] is not None and s["n"] >= cfg.ctx_min_seg_n:
                cand.append((s["id"], s["gmean"], s["gvar"]))
        if self.cur_x["n"] >= cfg.ctx_min_seg_n:
            n = self.cur_x["n"]
            m = self.cur_x["sum"] / n
            v = np.maximum(self.cur_x["sq"] / n - m * m, cfg.var_floor)
            cand.append((self.seg_counter, m, v))
        if not cand:
            return {self.seg_counter: 1.0}

        lls = np.array([self._diag_ll(x, m, v) for _, m, v in cand])
        w = np.exp((lls - lls.max()) / cfg.ctx_temp)
        w /= w.sum()
        w = (1.0 - cfg.ctx_uniform) * w + cfg.ctx_uniform / len(w)
        return {sid: float(wi) for (sid, _, _), wi in zip(cand, w)}


    # ---------------- ② 路由 + 响应 ----------------
    def _valid(self, rels):
        """合同有效的细胞：条件可靠性不低于门槛（冷启动时放宽为全体）。"""
        idx = [i for i in range(len(rels)) if rels[i] >= self.cfg.contract_floor]
        return idx if idx else list(range(len(rels)))

    def _select(self, dists, rels):
        cfg = self.cfg
        pool = self._valid(rels)
        pool_dists = [dists[i] for i in pool]
        pool_rels = rels[pool]
        best_local = int(np.argmax(pool_rels))
        best = pool[best_local]
        # 反射弧：条件可靠性最高的细胞高度自信 -> 单跳直达（毫秒级）
        if pool_rels[best_local] >= cfg.reflex_rel and pool_dists[best_local].max() >= cfg.reflex_conf:
            return [best], True
        k = min(cfg.k_max, len(pool) + 1)
        if cfg.router == "similarity":
            local = similarity_route(pool_dists, pool_rels, min(cfg.k_max, len(pool)))
            return [pool[i] for i in local], False
        # 分歧路由：锚点 = 全库熵最大的细胞（自由探索，保证任何细胞都有曝光机会），
        # 其余候选从合同有效的细胞中贪心选（受合同约束）。
        anchor = int(np.argmax([-float(np.sum(d * np.log(d + _EPS))) for d in dists]))
        return divergence_route(dists, k, anchor=anchor, pool=pool), False

    def decide(self, p, count_energy=True, ctx=None):
        cfg = self.cfg
        dists = [c.predict(p) for c in self.cells]
        if ctx is None:
            ctx = self._context(p)
        # 条件合同：只用"这个输入所在领地"上的可靠性投票
        rels = np.array([c.cond_reliability(ctx) for c in self.cells])
        selected, reflex = self._select(dists, rels)
        w = np.exp(cfg.ensemble_beta * rels[selected])
        w /= w.sum()
        ens = sum(wi * dists[i] for wi, i in zip(w, selected))
        pred = int(np.argmax(ens))
        conf = float(ens.max())
        if count_energy:
            # 能耗按"参数激活量"计：稀疏激活是小细胞相对大模型的结构性优势
            self.energy += sum(self.cells[i].n_params for i in selected)
        return pred, conf, selected, dists, ens, reflex, ctx

    # ---------------- ③ 观察 + 三因子学习 + 条件合同记账 ----------------
    def observe(self, p, y, pred, conf, selected, dists, ens, ctx):
        cfg = self.cfg
        correct = int(pred == y)
        surprise = float(-np.log(ens[y] + _EPS))
        dom = self.seg_counter      # 醒着只经历当下领地：功过记在当下

        # 三因子学习只发生在被选中的细胞（能量约束）；但**条件合同记账
        # 覆盖所有细胞**——每一步所有细胞都亮明了观点（dists 已全量算出），
        # 记账只是比对、零成本。合同因此忠实记录"该细胞在本领地观察到的
        # 履约率"，不论它是否被派活。
        # （实测教训：只给被选中者记账时，反射弧的亚军专才只在困难样本上
        # 露脸，合同收敛到困难样本精度 (~0.75) 再被惰性衰减拖到 0.39 而
        # 遭凋亡——而它的真实精度始终是 0.91。）
        for i, c in enumerate(self.cells):
            ok = int(np.argmax(dists[i]) == y)
            if i in selected:
                c.activations += 1
                c.correct += ok
                c.learn(p, y, cfg.lr if ok else cfg.lr_breach)
            c.note(dom, ok, cfg.reliability_alpha)
            c.seg_note[dom] = self.step

        # 当前概念段的运行统计（领地匹配 + 封存时的最终统计）
        cur = self.cur
        cur["sq"] += p * p
        cur["n"] += 1
        if y in cur["sums"]:
            cur["sums"][y] = cur["sums"][y] + p
            cur["ns"][y] += 1
        else:
            cur["sums"][y] = p.copy()
            cur["ns"][y] = 1

        # 解码空间的运行累加（当前段的分布模型）+ 信号 B（分布距离）：
        # 参照模型在段内冻结（否则模型跟着新任务漂移、距离被抹平），再用
        # 指数加权统计量（窗宽 ~1/ew_alpha 步）与参照比"分布距离"：
        # 逐维均值偏移（按方差归一）+ 逐维对数方差偏移。任务切换的落差
        # 13~90，稳态噪声 <0.7——比单样本似然更稳健：似然对"窄任务跟在
        # 宽任务后"不降反升（紧凑分布声称中心，实测线性->xor 漏检），
        # 而分布距离对两个方向都灵敏。
        x = self._decode(p)
        cx = self.cur_x
        cx["sum"] += x
        cx["sq"] += x * x
        cx["n"] += 1
        if self.seg_ref is None:
            if cx["n"] >= cfg.ctx_freeze_n:
                n = cx["n"]
                m = cx["sum"] / n
                v = np.maximum(cx["sq"] / n - m * m, cfg.var_floor)
                self.seg_ref = (m, v)
                self.ew = (m.copy(), v.copy())   # 从参照出发，只被数据推离
        else:
            m0, v0 = self.ew
            diff = x - m0
            m1 = m0 + cfg.ew_alpha * diff
            v1 = (1.0 - cfg.ew_alpha) * (v0 + cfg.ew_alpha * diff * diff)
            self.ew = (m1, np.maximum(v1, cfg.var_floor))
            mr, vr = self.seg_ref
            d = float(np.sum((m1 - mr) ** 2 / (vr + v1)
                             + (np.log(v1) - np.log(vr)) ** 2))
            if d > cfg.d_thr:
                if self.d_run == 0:
                    self.d_start = self.step   # 本段分布距离首次超限的步
                self.d_run += 1
            else:
                self.d_run = 0
                self.d_start = None

        # 经历缓冲：FIFO（保留时间序，封存时可按时间修剪边界滞后污染）
        self.buffer.append((p, y, surprise, self.step))
        if len(self.buffer) > cfg.buffer_size:
            self.buffer = self.buffer[-cfg.buffer_size:]

        # 做梦宽限期（新生儿扰动会造成短期精度抖动，不是真边界）
        if self.dream_grace > 0:
            self.dream_grace -= 1

        # 概念边界检测（双信号 OR，都要持续确认，滤除瞬时扰动）：
        #  A. 性能跌落：在线精度 EMA 相对历史峰值大幅跌落且**持续**——
        #     从"表现好"跌到"表现差"正是概念切换的定义；瞬时抖动（做梦扰动）
        #     几十步内自愈，持续跌落不会。峰值缓慢衰减防止旧峰值永存。
        #  B. 分布距离：当前段近期样本的解码统计量（指数加权）与段内冻结
        #     参照模型的差异超过阈值且持续（observe 里维护）。8 维统计量，
        #     任务切换落差 13~90 vs 稳态噪声 <0.7；对"精度还没掉但输入
        #     分布已经换了"的前几十步比信号 A 更灵敏，且不受做梦扰动影响
        #     （做梦不改变输入分布）。
        self.ema_acc = 0.9 * self.ema_acc + 0.1 * correct
        self.peak_acc = max(self.peak_acc * 0.9997, self.ema_acc)
        dropping = self.ema_acc < self.peak_acc - cfg.acc_drop
        if dropping and self.drop_run == 0:
            self.drop_start = self.step    # 本轮连续跌落的起点
        self.drop_run = self.drop_run + 1 if dropping else 0
        if not dropping:
            self.drop_start = None

        if (self.dream_grace == 0
                and self.step > cfg.warmup
                and self.step - self.last_seal > cfg.boundary_cooldown
                and (self.drop_run >= cfg.drop_sustain
                     or self.d_run >= cfg.d_sustain)):
            # 用"首次超限"而非"确认超限"作为段尾：持续确认有 15~25 步滞后，
            # 此间新任务的样本会污染本段 buffer 尾部（分布模型失准）
            starts = [t for t in (self.d_start, self.drop_start) if t is not None]
            self.boundary_step = min(starts) if starts else self.step
            self._seal_segment()
            self.peak_acc = self.ema_acc
            self.drop_run = 0
            self.d_run = 0

        return correct, surprise

    # ---------------- 封存概念段 ----------------
    def _seg_gauss(self, entries):
        """从 [(p, y, ...)] 计算解码空间的对角高斯 (mean, var)。

        鲁棒拟合：时间截断后仍可能残留少数跨段样本（边界检测的
        固有滞后），以全体估计为参照做一轮马氏距离离群剔除
        （χ²_8 的 99.9% 分位≈26，取 40 留余量），再重拟合——
        实测 8% 的污染样本可把 var 抬高一个量级。
        """
        xs = np.array([self._decode(p) for p, y, *_ in entries])
        m = xs.mean(axis=0)
        v = np.maximum(xs.var(axis=0), self.cfg.var_floor)
        keep = np.sum((xs - m) ** 2 / v, axis=1) < self.cfg.gauss_chi
        if keep.sum() >= 20:
            xs = xs[keep]
            m = xs.mean(axis=0)
            v = np.maximum(xs.var(axis=0), self.cfg.var_floor)
        return m, v

    def _seg_stats(self, entries):
        """从 [(p, y, ...)] 计算类别均值统计。"""
        sums, ns = {}, {}
        sq = np.zeros(self.pulse_dim)
        for p, y, *_ in entries:
            sq += p * p
            if y in sums:
                sums[y] = sums[y] + p
                ns[y] += 1
            else:
                sums[y] = p.copy()
                ns[y] = 1
        means = {c: sums[c] / ns[c] for c in sums if ns[c] >= 3}
        sigma = float(np.sqrt(np.mean(sq / max(len(entries), 1))))
        return dict(means=means, sigma=max(sigma, 1e-3), n=len(entries))

    def _seal_segment(self):
        """封存当前概念段：按时间修剪 + 类别分层保留高意外度片段。

        段级合并：如果新段与某个旧段的类别均值高度同向同尺，说明这是
        **老任务复现**，把新经历并入旧段（同一领地汇合），而不是新建一个
        谁也看不懂的段 —— 对应设计文档中"细胞合并"在段粒度上的实现。
        """
        cfg = self.cfg
        if len(self.buffer) < 20:
            return
        # 按时间丢弃最旧的一部分：边界检测有滞后，段头混着上一个任务
        entries = sorted(self.buffer, key=lambda t: t[3])
        # 尾部截断：边界检测同样有 15~25 步确认滞后，触发步之后的样本
        # 已经属于下一段，却落在本段 buffer 最新端（buf_trim 只砍旧端
        # 砍不到）。实测不截断时：21 个新段样本把 wave 的 gvar 从 0.36
        # 抬到 0.78、linear 前两维从 1.0 抬到 1.34——领地匹配的区分度
        # 被系统性破坏。
        if self.boundary_step is not None:
            entries = [t for t in entries if t[3] <= self.boundary_step]
            if len(entries) < 20:
                return
        cut = int(len(entries) * cfg.buf_trim)
        entries = entries[cut:]
        if len(entries) < 20:
            return
        stats = self._seg_stats(entries)
        # 解码空间的分布统计（领地匹配用）：对**整个修剪窗口**拟合，
        # 不用分层保留的高意外度子集——后者偏向边界附近、方差被系统性
        # 压低，与当前段的运行模型（全量缓冲）不可比。
        gstats = self._seg_gauss(entries)

        # 2) 段级合并检查：类别集合一致 + 各类均值同向（余弦）+ 同尺（模长比）
        for s in self.segments:
            if set(stats["means"]) != set(s["means"]):
                continue
            ok = True
            for c, m in stats["means"].items():
                mo = s["means"][c]
                cos = float(self._unit(m) @ self._unit(mo))
                nr = np.linalg.norm(m) / max(np.linalg.norm(mo), 1e-9)
                if cos < cfg.seg_merge_thr or not (1.0 / cfg.seg_merge_norm_tol
                                                   <= nr <= cfg.seg_merge_norm_tol):
                    ok = False
                    break
            if not ok:
                continue
            # 合并：样品汇流，统计按 0.7/0.3 融合，合同折写进旧段
            s["samples"].extend([(p, y) for p, y, _, _ in entries])
            if len(s["samples"]) > cfg.seg_merge_size:
                s["samples"] = s["samples"][-cfg.seg_merge_size:]
            for c, m in stats["means"].items():
                s["means"][c] = 0.7 * s["means"][c] + 0.3 * m
            s["sigma"] = 0.7 * s["sigma"] + 0.3 * stats["sigma"]
            s["n"] += stats["n"]
            # 分布模型同样按 0.7/0.3 融合（均值直接加权，方差用合并方差公式）
            mg, vg = gstats
            s["gmean"] = 0.7 * s["gmean"] + 0.3 * mg
            s["gvar"] = (0.7 * s["gvar"] + 0.3 * vg
                         + 0.21 * (s["gmean"] - mg) ** 2)
            for cell in self.cells:
                v = cell.seg_rel.pop(self.seg_counter, None)
                if v is not None and self.seg_counter != s["id"]:
                    old = cell.seg_rel.get(s["id"])
                    cell.seg_rel[s["id"]] = (0.7 * old + 0.3 * v) if old is not None else v
                    cell.seg_note.pop(self.seg_counter, None)
            self.seg_counter += 1     # 已复现的段号退役，不复用
            self._reset_current()
            self.events.append((self.step, "seal",
                                f"任务复现：{len(entries)} 个片段并入概念段 "
                                f"#{s['id']}（库存 {len(s['samples'])}）"))
            return

        # 3) 真正的新段：封存前把所有细胞的当前段合同冻结
        for cell in self.cells:
            cell.freeze(self.seg_counter, self.step, cfg.inactivity_decay)

        # 4) 按类别分层保留高意外度片段（信息瓶颈）
        by_class = {}
        for p, y, s, _ in entries:
            by_class.setdefault(y, []).append((s, p, y))
        per_class = max(8, cfg.seg_buffer_size // max(1, len(by_class)))
        seg = []
        for items in by_class.values():
            items.sort(key=lambda t: t[0], reverse=True)
            seg.extend([(p, yy) for _, p, yy in items[:per_class]])

        self.segments.append({"id": self.seg_counter, "samples": seg,
                              "means": stats["means"], "sigma": stats["sigma"],
                              "n": stats["n"],
                              "gmean": gstats[0], "gvar": gstats[1]})
        new_id = self.seg_counter
        self.seg_counter += 1
        if len(self.segments) > cfg.max_segments:
            dropped = self.segments.pop(0)   # 最老的段被压缩遗忘
            self.events.append((self.step, "forget",
                                f"最老概念段 #{dropped['id']} 被挤出"))
        self._reset_current()
        self.events.append((self.step, "seal",
                            f"概念段 #{new_id} 封存：保留 {len(seg)} 个高意外度"
                            f"片段（分层均衡），库存 {len(self.segments)} 段"))

    def _reset_current(self):
        self.buffer = []
        self.cur = dict(sums={}, ns={}, sq=np.zeros(self.pulse_dim), n=0)
        self.cur_x = dict(sum=np.zeros(self.cfg.d_in),
                          sq=np.zeros(self.cfg.d_in), n=0)
        self.seg_ref = None
        self.ew = None
        self.d_run = 0
        self.d_start = None
        self.drop_start = None
        self.boundary_step = None
        self.last_seal = self.step

    # ---------------- ④ 梦境代谢 ----------------
    def _pools(self):
        """重放池：[(segment_id, samples), ...]，含当前在流的段。"""
        pools = [(s["id"], s["samples"]) for s in self.segments]
        if self.buffer:
            pools.append((self.seg_counter, [(p, y) for p, y, _, _ in self.buffer]))
        return pools

    def _cell_loss(self, p, y):
        _, _, _, _, ens, _, _ = self.decide(p, count_energy=False)
        return -np.log(ens[y] + _EPS)

    def _ensemble_acc(self, samples):
        ok = 0
        for p, y in samples:
            pred, _, _, _, _, _, _ = self.decide(p, count_energy=False)
            ok += int(pred == y)
        return ok / max(len(samples), 1)

    def _rehearse(self, pools):
        """一次反刍批次：路由选细胞 -> 三因子巩固 + 条件合同记账。

        梦的优势：反刍样本自带段号元数据，无需推断领地，直接核对旧账。
        """
        cfg = self.cfg
        rng = self.rng
        batch = []
        for _ in range(cfg.dream_batch):
            seg_id, samples = pools[rng.integers(len(pools))]
            p, y = samples[rng.integers(len(samples))]
            batch.append((seg_id, p, y))

        def consolidate(seg_id, p, y):
            ctx = {seg_id: 1.0}
            dists = [c.predict(p) for c in self.cells]
            rels = np.array([c.cond_reliability(ctx) for c in self.cells])
            selected, _ = self._select(dists, rels)
            for i in selected:
                c = self.cells[i]
                ok = int(np.argmax(dists[i]) == y)
                c.learn(p, y, cfg.dream_lr if ok else cfg.lr_breach)
                c.note(seg_id, ok, cfg.dream_alpha)
                c.seg_note[seg_id] = self.step

        for seg_id, p, y in batch:
            consolidate(seg_id, p, y)
        # 反事实模拟：对最困难的片段自生成"如果当时……"的考题，
        # 只有在模拟考上履约的细胞才巩固（自生成考题通过才记入长期记忆）。
        if cfg.counterfactual:
            hard = sorted(batch, key=lambda t: -self._cell_loss(t[1], t[2]))
            for seg_id, p, y in hard[:max(1, len(batch) // 4)]:
                for _ in range(2):
                    consolidate(seg_id, p + cfg.cf_noise * rng.normal(size=p.shape), y)

    def dream(self):
        """离线梦境：反刍(+反事实) -> 凋亡 -> 蒸馏(分裂) -> (合并, 预留)。"""
        cfg = self.cfg
        pools = self._pools()
        if not pools:
            return
        rng = self.rng
        for _ in range(cfg.dream_batches):
            self._rehearse(pools)
        self.dream_grace = cfg.dream_grace

        # ---- 代谢：凋亡 -> 蒸馏 ----
        eval_samples = []
        for _, samples in pools:
            eval_samples.extend(samples[:24])
        ens_acc = self._ensemble_acc(eval_samples)

        # 凋亡判据 = 最佳领地可靠性（niche）。
        # 条件合同下，"在别人的任务上差"不再是罪——专才就该专；
        # 只有**在任何领地都没能建立可靠合同**的细胞才溶解回收。
        for idx in reversed(range(len(self.cells))):
            if len(self.cells) <= 2:
                break
            c = self.cells[idx]
            if c.niche() < cfg.prune_abs:
                dead = self.cells.pop(idx)
                self.events.append((self.step, "death",
                                    f"细胞 #{dead.cell_id} 凋亡（最佳领地可靠性 "
                                    f"{c.niche():.2f}，从未在任何任务上可靠），"
                                    f"容量回收，剩余 {len(self.cells)} 个"))

        # 蒸馏：集成在重放池上仍不达标 -> 分裂重放损失最大的细胞
        if ens_acc < cfg.split_acc and len(self.cells) < cfg.max_cells:
            worst, worst_loss = 0, -1.0
            for i, c in enumerate(self.cells):
                l = float(np.mean([-np.log(c.predict(p)[y] + _EPS)
                                   for p, y in eval_samples[:64]]))
                if l > worst_loss:
                    worst, worst_loss = i, l
            parent = self.cells[worst]
            child = parent.spawn(rng, cfg.split_noise, self.next_id, self.step)
            self.next_id += 1
            self.cells.append(child)
            self.events.append((self.step, "birth",
                                f"细胞 #{child.cell_id} 由 #{parent.cell_id} 分裂诞生"
                                f"（重放损失 {worst_loss:.2f} 过载），"
                                f"库存 {len(self.cells)}/{cfg.max_cells}"))

        # 细胞级合并（预留机制，MVP 默认关闭）
        if cfg.merge_enabled and len(self.cells) > 2:
            best_pair, best_agree = None, 0.0
            for i in range(len(self.cells)):
                for j in range(i + 1, len(self.cells)):
                    agree = float(np.mean([
                        np.argmax(self.cells[i].predict(p)) == np.argmax(self.cells[j].predict(p))
                        for p, _ in eval_samples[:64]]))
                    if agree > best_agree:
                        best_pair, best_agree = (i, j), agree
            if best_pair is not None and best_agree > 0.99:
                i, j = best_pair
                loser = i if self.cells[i].niche() < self.cells[j].niche() else j
                dead = self.cells.pop(loser)
                self.events.append((self.step, "merge",
                                    f"细胞 #{dead.cell_id} 与邻居合并（一致率 {best_agree:.2f}）"))

    # ---------------- 对外接口 ----------------
    def run_step(self, x, y):
        """推进一个样本的完整生命周期，返回本步指标。"""
        # 领地惰性衰减：当前段活跃，久不履约的细胞合同贬值（封存的段已冻结）
        for c in self.cells:
            c.erode(self.seg_counter, self.step, self.cfg.inactivity_decay)
        p = self.perceive(x)
        pred, conf, selected, dists, ens, reflex, ctx = self.decide(p, count_energy=True)
        correct, surprise = self.observe(p, y, pred, conf, selected, dists, ens, ctx)
        if self.cfg.dream_enabled and self.step > 0 and self.step % self.cfg.dream_period == 0:
            self.dream()
        rec = dict(step=self.step, correct=correct, surprise=surprise,
                   k=len(selected), reflex=int(reflex),
                   n_cells=len(self.cells),
                   n_params=sum(c.n_params for c in self.cells),
                   energy=self.energy)
        self.history.append(rec)
        self.step += 1
        return rec

    def evaluate(self, test_sets):
        """在所有见过的任务的测试集上评估（保持率指标；不计能耗）。

        条件合同在此真正接受检验：系统看到的只是裸输入，
        必须靠类别均值匹配自己推断"这是哪个任务的地盘"。
        """
        accs = []
        for X, y in test_sets:
            ok = sum(int(self.decide(self.perceive(x), count_energy=False)[0] == yy)
                     for x, yy in zip(X, y))
            accs.append(ok / len(y))
        return accs

    def territory_accuracy(self, test_sets, seg_of_task):
        """领地推断准确率：每个任务的测试输入，其匹配度最高的段
        是否就是该任务对应的段（用于检验上下文推断的质量）。"""
        ok, tot = 0, 0
        for ti, (X, y) in enumerate(test_sets):
            want = seg_of_task[ti]
            for x in X:
                p = self.perceive(x)
                ctx = self._context(p)
                got = max(ctx.items(), key=lambda kv: kv[1])[0]
                ok += int(got == want)
                tot += 1
        return ok / max(tot, 1)

    @property
    def n_params(self):
        return sum(c.n_params for c in self.cells)
