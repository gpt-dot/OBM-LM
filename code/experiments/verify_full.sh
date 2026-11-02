#!/bin/bash
# verify_full.sh —— 6 段完整验证（4 域 + 2 复现，1800 步，与 pilot_v1 同协议）
# 验证：① 完整功能（seal/birth/death 全点火）② 效果（ppl 对标 pilot_v1）
# ctx128（最快）与 b12（全长上下文）两个候选都验，按效果+速度择优。
cd "$(cd "$(dirname "$0")" && pwd)/.."
export MALLOC_ARENA_MAX=2 OBM_THREADS=8
PY=python3

echo "########## FULL VERIFY START $(date) ##########"

# 候选1：ctx128（最快 2.35x）
echo "=== verify_ctx128 (6-block, eval) ==="
$PY experiments/obm_lab.py --tag verify_ctx128 --d-model 256 --n-layers 4 \
    --n-heads 8 --batch 24 --ctx 128 --n-blocks 6 --steps-per-domain 300 \
    --eval --dist-refresh 10

# 候选2：b12（全长上下文 256，效果更稳）
echo "=== verify_b12 (6-block, eval) ==="
$PY experiments/obm_lab.py --tag verify_b12 --d-model 256 --n-layers 4 \
    --n-heads 8 --batch 12 --ctx 256 --n-blocks 6 --steps-per-domain 300 \
    --eval --dist-refresh 10

echo "=== report ==="
$PY experiments/report_lab.py
echo "########## FULL VERIFY DONE $(date) ##########"
