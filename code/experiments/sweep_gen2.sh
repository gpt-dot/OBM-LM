#!/bin/bash
# sweep_gen2.sh —— Gen2 手术后的杠杆扫描：减小每步位置数 B×L 三方向
# 串行执行（避免两训练进程 OOM）。每个 3-block×200 步，验 seal 功能 + 测速。
cd "$(cd "$(dirname "$0")" && pwd)/.."
export MALLOC_ARENA_MAX=2 OBM_THREADS=8
PY=python3
LAB=experiments/obm_lab.py
COMMON="--n-blocks 3 --steps-per-domain 200 --dist-refresh 10"

echo "########## SWEEP START $(date) ##########"

# A) batch 减半 24->12：每步位置减半，backward/前向/路由全线性受益
$PY $LAB --tag g2_b12   --d-model 256 --n-layers 4 --n-heads 8 --batch 12 $COMMON
# B) ctx 减半 256->128：位置减半，保持 batch
$PY $LAB --tag g2_ctx128 --d-model 256 --n-layers 4 --n-heads 8 --batch 24 --ctx 128 $COMMON
# C) 中间尺寸 d192L3：参数量与 V 维负载折中
$PY $LAB --tag g2_d192  --d-model 192 --n-layers 3 --n-heads 6 --batch 24 $COMMON

echo "########## SWEEP DONE $(date) ##########"
