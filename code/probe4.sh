#!/bin/bash
echo "===== PROBE4 @ $(date) ====="
echo "--- load apps/PyTorch/1.10/dtk-22.04-py37 (no pipe!) ---"
module load apps/PyTorch/1.10/dtk-22.04-py37 2>&1
echo "exit=$?"
echo "PATH prefix: $(echo $PATH | cut -c1-300)"
PY3=$(command -v python3)
echo "python3 = $PY3"
$PY3 --version 2>&1
echo "--- import torch ---"
$PY3 -c "import torch; print('torch', torch.__version__); print('threads', torch.get_num_threads())" 2>&1 | head -5
echo "--- import numpy ---"
$PY3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | head -3
echo "--- mini matmul bench (32f x 32f) ---"
$PY3 -c "
import torch, time
a=torch.randn(1024,1024); b=torch.randn(1024,1024)
torch.set_num_threads(32)
t=time.perf_counter()
for _ in range(10): c=a@b
dt=time.perf_counter()-t
print('10x 1024^3 matmul: %.2fs => %.1f GFLOPS' % (dt, 20*1024**3/dt/1e9))
" 2>&1 | head -6
echo "PROBE4 done"
