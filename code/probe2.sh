#!/bin/bash
echo "===== PROBE2 @ $(date) ====="
echo "--- load python/3.8.10 ---"
module load python/3.8.10 2>&1 | head -3
which python python3 2>/dev/null
python3 --version 2>&1
echo "--- internet check (real) ---"
python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org',timeout=8); print('pypi OK')" 2>&1
echo "--- search existing torch on fs ---"
find /public/software -maxdepth 7 -name "torch" -type d 2>/dev/null | head
ls /public/software/python/3.8.10/lib/python3.8/site-packages 2>/dev/null | grep -i torch
echo "--- try pip install torch cpu (180s budget) ---"
timeout 180 python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -6 || echo "pip torch failed/timeout"
python3 -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())" 2>&1 | head -3
echo "PROBE2 done"
