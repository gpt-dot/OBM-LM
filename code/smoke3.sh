#!/bin/bash
REMOTE_HOME="${REMOTE_HOME:-/public/home/your_account}"
set -o pipefail
echo "===== SMOKE3B @ $(date) ====="
module load apps/PyTorch/1.10/dtk-22.04-py37 2>/dev/null || true
PYBIN=$(command -v python3)
echo "PYBIN=$PYBIN $("$PYBIN" --version 2>&1)"
export OBM_ROOT=${REMOTE_HOME}/obm_mvp
export OBM_THREADS=32
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

# --- scale2 级别 500 步实测速度 ---
/usr/bin/time -v "$PYBIN" -X faulthandler experiments/obm_lab.py \
  --tag smoke3 \
  --d-model 384 --n-layers 6 --n-heads 6 \
  --batch 16 --ctx 128 \
  --n-cells 4 --max-cells 6 \
  --lr 1.2e-3 \
  --steps-per-domain 250 --n-blocks 2 \
  --dist-refresh 10 --dream-period 200 \
  --no-amp \
  --weights-dir "$OBM_ROOT/results_lab" --weights-every 250 \
  2>&1 | tee results_lab/smoke3.log
echo "EXIT=$?"
echo "===== SMOKE3B done @ $(date) ====="
