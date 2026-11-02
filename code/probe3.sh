#!/bin/bash
echo "===== PROBE3 @ $(date) ====="
echo "--- module load apps/python/3.8.13 ---"
module load apps/python/3.8.13 2>&1 | head -2
which python python3 pip 2>/dev/null
python3 --version 2>&1; python --version 2>&1
echo "--- module load apps/PyTorch/1.10/dtk-22.04-py37 ---"
module purge 2>/dev/null
module load apps/PyTorch/1.10/dtk-22.04-py37 2>&1 | head -2
which python python3 2>/dev/null
PYBIN=$(which python3 2>/dev/null || which python 2>/dev/null)
echo "PYBIN=$PYBIN"
$PYBIN --version 2>&1
$PYBIN -c "import torch; print('torch', torch.__version__)" 2>&1 | head -3
$PYBIN -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | head -2
echo "PROBE3 done"
