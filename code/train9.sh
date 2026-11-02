#!/bin/bash
REMOTE_HOME="${REMOTE_HOME:-/public/home/your_account}"
# scale9 —— 数据新鲜度主攻轮（头脑风暴+TRIZ 第十轮定稿）
# 新语料 146.9M tokens（wiki 翻倍 78M，tok/param 0.65→1.94）
# TRIZ 组合：No.35 参数变化（新数据杠杆）+ No.23 反馈（eval-every 闭环）
#          + No.5 动态化（dream-seq 序列梦 A/B）+ No.10 预先作用（spawn-cells 补第4细胞）
set -x
cd ${REMOTE_HOME}/obm_mvp
export OBM_THREADS=30 OBM_ROOT=${REMOTE_HOME}/obm_mvp OBM_WD_FP16=1
export MALLOC_ARENA_MAX=2
ulimit -c 0

# 离线 torch 1.13.1+cpu（dtk 版 autograd 在无 GPU 节点 SIGABRT 的绕行）
mkdir -p /tmp/torchcpu
[ -d /tmp/torchcpu/torch ] || unzip -o -q torchcpu.zip -d /tmp/torchcpu
export PYTHONPATH=/tmp/torchcpu:$PYTHONPATH
python3 -c "import torch; assert torch.__version__.startswith('1.13')" || exit 9

# resume 链：scale8 -> scale7 回退
R=results_lab
SRC=""
for c in "$R/scale8_weights.pkl" "$R/scale7_weights.pkl"; do
  [ -f "$c" ] && SRC="$c" && break
done
[ -n "$SRC" ] || { echo "no resume source"; exit 8; }

RESUME_ARGS=""
if [ -n "$SRC" ]; then
  cp "$SRC" ./resume_src.pkl
  RESUME_ARGS="--resume ./resume_src.pkl --spawn-cells 1"
fi

# 4 域对称 4x2500=10000 步；replay-need + eval-every 真信号闭环；序列梦
python3 experiments/obm_lab.py \
  --tag scale9 \
  --d-model 512 --n-layers 8 --batch 16 --ctx 128 \
  --n-cells 3 --max-cells 4 \
  --steps-per-domain 2500 --n-blocks 4 \
  --domains en_lit,zh_classic,code_py,en_wiki \
  --replay 0.3 --replay-need \
  --dream-seq \
  --eval-every 500 \
  --lr 4e-4 \
  $RESUME_ARGS \
  --weights-every 1000 --eval \
  2>&1 | tee "$R/scale9.log"
