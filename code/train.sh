#!/bin/bash
REMOTE_HOME="${REMOTE_HOME:-/public/home/your_account}"
set -o pipefail
echo "===== OBM-LM SCALE4 training @ $(date) ====="
module load apps/PyTorch/1.10/dtk-22.04-py37 2>/dev/null || true
PYBIN=$(command -v python3)
echo "PYBIN=$PYBIN $("$PYBIN" --version 2>&1)"
export OBM_ROOT=${REMOTE_HOME}/obm_mvp
export OBM_THREADS=32
export OBM_WD_FP16=1
export MALLOC_ARENA_MAX=2
cd "$OBM_ROOT"
ulimit -c 0
mkdir -p results_lab

# --- 离线安装真 CPU torch（dtk build 的 autograd 在无 GPU 节点 SIGABRT，弃用）---
rm -rf /tmp/torchcpu && mkdir -p /tmp/torchcpu
unzip -q -o ${REMOTE_HOME}/wheels/torch-1.13.1+cpu-cp37-cp37m-linux_x86_64.whl -d /tmp/torchcpu
unzip -q -o ${REMOTE_HOME}/wheels/typing_extensions-4.7.1-py3-none-any.whl -d /tmp/torchcpu
export PYTHONPATH=/tmp/torchcpu:$PYTHONPATH
$PYBIN -c "
import torch
assert '1.13' in torch.__version__, 'wrong torch: %s' % torch.__version__
x = torch.randn(16, 16, requires_grad=True); (x @ x).sum().backward()
assert x.grad is not None
print('[env] torch', torch.__version__, 'backward OK, threads=', torch.get_num_threads())
" || { echo "[env] FATAL: CPU torch broken"; exit 2; }

WEIGHTS="$OBM_ROOT/results_lab/scale4_weights.pkl"
RESUME=""
[ -f "$WEIGHTS" ] && RESUME="--resume $WEIGHTS"
echo "[train] resume flag: ${RESUME:-<fresh>}"

# --- scale4：smoke3b 实测 1.43s/step @53.74M；本配置 151M->302M，12000 步 ~15h ---
"$PYBIN" -X faulthandler experiments/obm_lab.py \
  --tag scale4 \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --batch 16 --ctx 128 \
  --n-cells 6 --max-cells 12 \
  --lr 1.2e-3 \
  --steps-per-domain 2000 --n-blocks 6 \
  --dist-refresh 10 --dream-period 200 \
  --no-amp \
  --weights-dir "$OBM_ROOT/results_lab" --weights-every 500 \
  --eval $RESUME 2>&1 | tee results_lab/scale4.log

# 终局权重复制到 home 根目录，便于下载
cp -f "$WEIGHTS" "$OBM_ROOT/scale4_weights.pkl" 2>/dev/null || true
echo "===== training done @ $(date) ====="
