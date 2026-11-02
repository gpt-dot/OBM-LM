#!/bin/bash
# 探测计算节点可用的 python / torch 模块（无外网，必须走 module）。
echo "===== PROBE @ $(date) ====="
echo "hostname=$(hostname) nproc=$(nproc)"
echo "--- default python ---"
which python python3 2>/dev/null || echo "NO python/python3 in PATH"

echo "--- module avail: python/torch/conda ---"
module avail 2>&1 | grep -iE "python|torch|anaconda|conda|pytorch|numpy" || echo "  (no match in avail list)"

echo "--- try loading candidate modules ---"
for m in python/3.11 python/3.10 python3 anaconda3 anaconda/3 anaconda pytorch torch torch/2.1 torch/2.0 cuda; do
  echo "### module load $m"
  module load "$m" 2>&1 | head -2
  if command -v python3 >/dev/null 2>&1; then
    echo "  python3 -> $(python3 --version 2>&1)"
    python3 -c "import torch,numpy; print('  torch', torch.__version__, 'numpy', numpy.__version__)" 2>&1 | head -1
  else
    echo "  still no python3"
  fi
done

echo "--- search python3 binaries under /public/software ---"
find /public/software -maxdepth 4 -name "python3" -type f 2>/dev/null | head -10
echo "--- conda envs ---"
ls /public/software/anaconda*/envs 2>/dev/null | head
echo "===== PROBE done ====="
