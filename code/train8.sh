#!/bin/bash
REMOTE_HOME="${REMOTE_HOME:-/public/home/your_account}"
set -o pipefail
echo "===== OBM-LM SCALE8 (long balanced + NLL-router eval) @ $(date) ====="
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

# --- 离线安装真 CPU torch ---
rm -rf /tmp/torchcpu && mkdir -p /tmp/torchcpu
unzip -q -o ${REMOTE_HOME}/wheels/torch-1.13.1+cpu-cp37-cp37m-linux_x86_64.whl -d /tmp/torchcpu
unzip -q -o ${REMOTE_HOME}/wheels/typing_extensions-4.7.1-py3-none-any.whl -d /tmp/torchcpu
export PYTHONPATH=/tmp/torchcpu:$PYTHONPATH
$PYBIN -c "
import torch
assert '1.13' in torch.__version__, 'wrong torch: %s' % torch.__version__
x = torch.randn(16, 16, requires_grad=True); (x @ x).sum().backward()
print('[env] torch', torch.__version__, 'backward OK')
" || { echo "[env] FATAL: CPU torch broken"; exit 2; }

# --- resume 源：scale7 优先，缺则回退 scale6 ---
if [ -f "$OBM_ROOT/results_lab/scale7_weights.pkl" ]; then
  SRC="$OBM_ROOT/results_lab/scale7_weights.pkl"
else
  SRC="$OBM_ROOT/scale6_weights.pkl"
fi
echo "[train] resume from $SRC"

# --- scale8：对称四域 4 x 1500 步 ≈ 4.5h；lr 4e-4 续退火 ---
$PYBIN -X faulthandler experiments/obm_lab.py \
  --tag scale8 \
  --d-model 512 --n-layers 8 --n-heads 8 \
  --batch 16 --ctx 128 \
  --n-cells 3 --max-cells 8 \
  --lr 4e-4 \
  --domains en_lit,zh_classic,code_py,en_wiki \
  --replay 0.25 --replay-lru \
  --steps-per-domain 1500 --n-blocks 4 \
  --dist-refresh 10 --dream-period 200 \
  --no-amp \
  --weights-dir "$OBM_ROOT/results_lab" --weights-every 500 \
  --resume "$SRC" \
  --eval --eval-router 2>&1 | tee results_lab/scale8.log

cp -f "$OBM_ROOT/results_lab/scale8_weights.pkl" "$OBM_ROOT/scale8_weights.pkl" 2>/dev/null || true
echo "===== training done @ $(date) ====="
