"""路由模块 —— OBM 的核心创新点 M2：分歧驱动路由。

传统 MoE 按"相似度/置信度"选专家（把输入送给最懂它的专家）；
OBM 反过来按"分歧度"选：把输入同时送给一群互相冲突的细胞。

直觉：分歧 = 系统对自己的预测不一致 = 意外 = 信息量最大。
只有让冲突的细胞同台，才能真正定位"我错在哪里"。
"""
import numpy as np

_EPS = 1e-12


def kl(p, q):
    """KL(p || q)，衡量两个细胞"观点"的距离。"""
    return float(np.sum(p * np.log((p + _EPS) / (q + _EPS))))


def divergence_route(dists, k, anchor=None, pool=None, strict_pool=False):
    """贪心选出分歧最大的 k 个细胞。

    策略：先选熵最大的细胞作锚点（最不确定 = 最可能出错 = 最值得学），
    再迭代加入"与已选集合平均 KL 最大"的细胞。

    anchor: 指定锚点下标（None 时自动取全库熵最大者）。
            OBM 传入全库锚点：保证任何细胞——包括可靠性衰退的旧专家——
            都有被激活的机会，避免"旧知识永不复兴"的死锁。
    pool:   候选集合（None 时为全库）。OBM 传入合同有效的细胞，
            即"锚点自由探索 + 候选受合同约束"。
    strict_pool: True 时 pool 升级为硬约束——k >= n 的早返回只回池内细胞，
            且传入的池外锚点被改指到池内熵最大者。LM 移植专用：实测
            锚点越池会把旧专才拉去训练异任务（copy 专才被 reverse 化
            的主通道），专才保护优先于曝光机会；LM 的旧专家复兴走
            "条件可靠性重回池 + 窗口期 probe 授予"两条通道。
    """
    n = len(dists)
    if k >= n:
        if strict_pool and pool is not None:
            return list(pool)
        return list(range(n))
    entropy = [-float(np.sum(d * np.log(d + _EPS))) for d in dists]
    if anchor is None:
        anchor = int(np.argmax(entropy))
    elif strict_pool and pool is not None and anchor not in pool:
        anchor = int(pool[int(np.argmax([entropy[j] for j in pool]))])
    selected = [anchor]
    candidates = [j for j in (pool if pool is not None else range(n)) if j != anchor]
    while len(selected) < k and candidates:
        best_idx, best_score = -1, -1.0
        for j in candidates:
            score = float(np.mean([kl(dists[j], dists[i]) for i in selected]))
            if score > best_score:
                best_idx, best_score = j, score
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


def similarity_route(dists, reliabilities, k):
    """基线路由（消融实验用）：按合同可靠性选 top-k。

    等价于"永远请教你最信任的老专家"——MoE 式门控。
    问题：老专家只对旧任务可靠，新任务与任务复现时无人可问。
    """
    order = np.argsort(reliabilities)[::-1]
    return [int(i) for i in order[:k]]
