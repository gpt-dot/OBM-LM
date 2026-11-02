#!/bin/bash
# verify_best.sh —— 等 sweep 完成 → 选最快且封存点火的配置 → 6 段完整验证（功能+效果）→ 报告
cd "$(cd "$(dirname "$0")" && pwd)/.."
export MALLOC_ARENA_MAX=2 OBM_THREADS=8
PY=python3
SWEEP_PID=347498

echo "=== waiting for sweep (pid $SWEEP_PID) ==="
while ps -p $SWEEP_PID >/dev/null 2>&1; do sleep 15; done
echo "=== sweep done $(date) ==="
$PY experiments/report_lab.py

# 选最快且 seal>=1（功能合格）的配置
read TAG D L B CTX <<< $($PY -c "
import json,glob
rows=[json.load(open(f)) for f in glob.glob('results_lab/g2_*.json')]
rows=[r for r in rows if r.get('event_counts',{}).get('seal',0)>=1]
if not rows:
    print('NONE 0 0 0 0'); raise SystemExit
rows.sort(key=lambda r:r['step_time_mean'])
b=rows[0]
print(b['tag'], b['d_model'], b['n_layers'], b['batch'], b.get('ctx',256))
")
echo "=== BEST passing config: tag=$TAG D=$D L=$L B=$B ctx=$CTX ==="
if [ "$TAG" = "NONE" ]; then echo "NO PASSING CONFIG"; exit 1; fi
H=$((D/32))
echo "=== 6-block full verification (4 domains + 2 reprise, eval ppl) ==="
$PY experiments/obm_lab.py --tag verify_$TAG --d-model $D --n-layers $L \
    --n-heads $H --batch $B --ctx $CTX --n-blocks 6 --steps-per-domain 300 \
    --eval --dist-refresh 10
echo "=== verify done $(date) ==="
$PY experiments/report_lab.py
