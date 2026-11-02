#!/usr/bin/env python3
"""打包 OBM-LM 全部永久产物为 zip。
- 代码：deploy/ 全部 + 顶层流水线/远端作业脚本 + obm_mvp（MVP与历史结果）
- 报告：11 份 HTML 战报/分析 + 训练方法 txt
- 模型：4 个训后权重 pkl（store 模式，浮点权重压缩无益）
排除：训前初始权重（输入而非产物）、缓存、pycache、日志。
"""
import os
import zipfile

WS = "/workspace"
OUT = os.path.join(WS, "OBM-LM-永久产物-20261101.zip")

MODELS = [
    "OBM-LM-scale4-训后-75M.pkl",
    "OBM-LM-scale5-训后-75M-v2.pkl",
    "OBM-LM-scale6-训后-75M.pkl",
    "OBM-LM-scale7-训后-75M.pkl",
]
REPORTS = [
    "OBM-LM-TRIZ矛盾解析与消解报告.html",
    "OBM-LM-TRIZ第二轮战报.html",
    "OBM-LM-TRIZ第三轮战报.html",
    "OBM-LM-TRIZ第四轮战报.html",
    "OBM-LM-头脑风暴xTRIZ第五轮战报.html",
    "OBM-LM-头脑风暴xTRIZ第六轮战报.html",
    "OBM-LM-头脑风暴xTRIZ第七轮战报.html",
    "OBM-LM-四引擎战报第八轮.html",
    "OBM-LM-四引擎战报第九轮.html",
    "OBM-LM-双引擎第十轮战报.html",
    "OBM-LM-跨代生成对比.html",
    "OBM-LM-训练方法.txt",
]
TOP_SCRIPTS = [
    "auto_chain.py", "night_pipeline.py",
    "scnet_client.py", "scnet_submit.py", "scnet_upload.py",
    "upload_corpus.py",
]
CODE_DIRS = ["deploy", "obm_mvp"]

SKIP_EXT = (".pyc", ".pyo", ".npy", ".pkl", ".parquet", ".bin", ".log")
SKIP_DIRS = {"__pycache__", ".git", "cache", "wheels", "node_modules"}

README = """OBM-LM 永久产物包（2026-11-01）
====================================

一、模型权重 models/（75M 参数 OBM 细胞市场 LM，各代训后）
  - scale4-训后: 第四代，146.9M token 语料路线的起点代
  - scale5-训后-v2: 第五代
  - scale6-训后: 第六代
  - scale7-训后: 第七代（当前最强，en_lit 120 / zh 837 / code 72 / wiki 293）

二、报告 reports/
  - TRIZ矛盾解析与消消报告 + 第二~十轮战报（TRIZ -> 头脑风暴xTRIZ -> 四引擎）
  - 跨代生成对比.html: scale4~7 同 prompt 生成质量对比
  - 训练方法.txt: 训练方法概要

三、代码 code/
  - deploy/: 远端作业用的完整推理+训练栈
      obm/          细胞市场核心（cell/mesh/router/stream/baselines）
      obm_lm/       语言模型层（autograd/cell_lm/corpus/pure_bpe/tokenizer...）
      experiments/  obm_lab.py（训练主控）、run_scale.py、demo_sandbox.py
      train*.sh     scale5~9 提交脚本（train9 = 待跑的接力配置）
  - obm_mvp/: MVP 阶段代码与历史结果
  - 顶层脚本: scnet_*（算力平台客户端）、auto_chain.py（队列套利）、
    night_pipeline.py（夜间流水线）、upload_corpus.py（语料上传）

四、复现要点
  1. NLL 路由推理: deploy/experiments/demo_sandbox.py --router nll-top1
     --weights "scale4.pkl,scale5.pkl,scale6.pkl,scale7.pkl"（zoo 模式，跨代合并池）
  2. 关键修复备忘: load_zoo 不做 cell_id 去重（跨代撞号会误删专才）;
     NLL 决策必须用无填充真实尾部 64 token; 新手细胞给全速 lr。
  3. 训练: 修改 deploy/train*.sh 后经 scnet_submit.py 提交（队列 Agent0）。
"""

def skip(name):
    parts = name.split("/")
    if any(p in SKIP_DIRS for p in parts):
        return True
    if name.endswith(SKIP_EXT):
        return True
    return False

added = 0
with zipfile.ZipFile(OUT, "w", allowZip64=True) as z:
    z.writestr("OBM-LM-永久产物/README.txt", README)
    # 模型（store）
    for m in MODELS:
        src = os.path.join(WS, m)
        z.write(src, f"OBM-LM-永久产物/models/{m}", compress_type=zipfile.ZIP_STORED)
        added += 1
        print("model:", m)
    # 报告（deflate）
    for r in REPORTS:
        src = os.path.join(WS, r)
        if not os.path.exists(src):
            print("!! 缺报告:", r); continue
        z.write(src, f"OBM-LM-永久产物/reports/{r}", compress_type=zipfile.ZIP_DEFLATED)
        added += 1
    # 顶层脚本
    for s in TOP_SCRIPTS:
        src = os.path.join(WS, s)
        z.write(src, f"OBM-LM-永久产物/code/{s}", compress_type=zipfile.ZIP_DEFLATED)
        added += 1
    # 代码目录（递归，跳过缓存/大文件）
    for d in CODE_DIRS:
        for root, dirs, files in os.walk(os.path.join(WS, d)):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                src = os.path.join(root, f)
                arc = f"OBM-LM-永久产物/code/{os.path.relpath(src, WS)}"
                if skip(arc):
                    continue
                z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED)
                added += 1

size = os.path.getsize(OUT)
print(f"\n完成: {OUT}")
print(f"条目数: {added}, 体积: {size/1024/1024:.1f} MB")
