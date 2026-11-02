#!/bin/bash
# verify2.sh —— ① b12 带分裂诊断（解释 birth 为何不点火）② ctx128 修好 eval 的完整验证
cd "$(cd "$(dirname "$0")" && pwd)/.."
export MALLOC_ARENA_MAX=2 OBM_THREADS=8
PY=python3
echo "########## VERIFY2 START $(date) ##########"

echo "=== verify_b12_dbg (split diagnostic ON) ==="
OBM_DEBUG_SPLIT=1 $PY experiments/obm_lab.py --tag verify_b12_dbg --d-model 256 \
    --n-layers 4 --n-heads 8 --batch 12 --ctx 256 --n-blocks 6 \
    --steps-per-domain 300 --eval --dist-refresh 10 2> results_lab/split_dbg.err

echo "=== verify_ctx128 (fixed eval) ==="
$PY experiments/obm_lab.py --tag verify_ctx128 --d-model 256 --n-layers 4 \
    --n-heads 8 --batch 24 --ctx 128 --n-blocks 6 --steps-per-domain 300 \
    --eval --dist-refresh 10

echo "=== report ==="
$PY experiments/report_lab.py
echo "########## VERIFY2 DONE $(date) ##########"
