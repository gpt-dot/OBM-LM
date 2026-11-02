#!/bin/bash
# SCNet 冒烟脚本：探测计算节点环境（Python/torch/核数/内存）。
# 由 SCNet 作业调度器直接执行；输出写入 stdout 供作业详情查看。
set -e
echo "===== SCNet compute-node smoke @ $(date) ====="
echo "hostname: $(hostname)"
echo "whoami : $(whoami)"
echo "nproc   : $(nproc)"
echo "uname   : $(uname -a)"
echo "--- python ---"
which python3 || echo "NO python3"
python3 --version 2>&1 || echo "python3 version failed"
echo "--- memory ---"
free -m 2>/dev/null || cat /proc/meminfo | head -3
echo "--- torch / numpy ---"
python3 -c "import torch,numpy; print('torch', torch.__version__, 'numpy', numpy.__version__); print('torch threads', torch.get_num_threads())" 2>&1 || echo "torch/numpy NOT available"
echo "--- module avail (if any) ---"
command -v module >/dev/null && module avail 2>&1 | head -20 || echo "no module cmd"
echo "--- internet check ---"
timeout 8 python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=6); print('pypi reachable')" 2>&1 || echo "pypi NOT reachable"
echo "===== smoke done ====="
