#!/bin/bash
# run_scale_chain.sh —— S1 完成后串行跑 S2/S3 的守护编排（守 8GB cgroup）。
# 用法：bash run_scale_chain.sh <S1_pid>
#   * 等 S1 进程退出
#   * 校验 S1 产出 metrics.json + 有机制事件（否则告警停链，避免浪费算力）
#   * S1 路由诊断 + 报告重生成
#   * 串行启动 S2、S3（各自独立 run_scale.py），每步 diag + report
#   * 全程日志到 results_scale/chain.log
set -u
cd "$(cd "$(dirname "$0")" && pwd)/.."
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=8
S1_PID="${1:?need S1 pid}"
LOG=results_scale/chain.log
ts() { date '+%m-%d %H:%M:%S'; }

echo "[$(ts)] chain start; waiting for S1 pid $S1_PID" | tee -a "$LOG"
while kill -0 "$S1_PID" 2>/dev/null; do sleep 30; done
echo "[$(ts)] S1 exited (pid $S1_PID gone)" | tee -a "$LOG"

# ---- S1 校验门控 ----
if [ ! -f results_scale/S1_v2/metrics.json ]; then
  echo "[$(ts)] ERROR: S1_v2/metrics.json missing — 停链" | tee -a "$LOG"
  exit 1
fi
NEV=$(python3 -c "import json;m=json.load(open('results_scale/S1_v2/metrics.json'));print(len(m.get('events_all',[])))" 2>/dev/null || echo 0)
echo "[$(ts)] S1 events_all count = $NEV" | tee -a "$LOG"
if [ "${NEV:-0}" -lt 1 ]; then
  echo "[$(ts)] WARN: S1 无机制事件，疑似异常 — 停链待人工检查" | tee -a "$LOG"
  exit 2
fi

# ---- S1 诊断 + 报告 ----
echo "[$(ts)] S1 diag_routing + report" | tee -a "$LOG"
python3 experiments/diag_routing.py --scale S1 --ckpt results_scale/S1_v2/ckpt_final.pkl >> "$LOG" 2>&1 || echo "[$(ts)] S1 diag failed (non-fatal)" | tee -a "$LOG"
python3 experiments/report_scale.py >> "$LOG" 2>&1 || echo "[$(ts)] report failed" | tee -a "$LOG"

# ---- S2 ----
echo "[$(ts)] launching S2" | tee -a "$LOG"
python3 experiments/run_scale.py --scale S2 --tag _v2 --seed 0 > results_scale/S2_v2.log 2>&1
echo "[$(ts)] S2 done (exit $?)" | tee -a "$LOG"
if [ -f results_scale/S2_v2/metrics.json ]; then
  python3 experiments/diag_routing.py --scale S2 --ckpt results_scale/S2_v2/ckpt_final.pkl >> "$LOG" 2>&1 || echo "[$(ts)] S2 diag failed (non-fatal)" | tee -a "$LOG"
  python3 experiments/report_scale.py >> "$LOG" 2>&1 || echo "[$(ts)] report failed" | tee -a "$LOG"
else
  echo "[$(ts)] WARN: S2 metrics.json missing" | tee -a "$LOG"
fi

# ---- S3 ----
echo "[$(ts)] launching S3" | tee -a "$LOG"
python3 experiments/run_scale.py --scale S3 --tag _v2 --seed 0 > results_scale/S3_v2.log 2>&1
echo "[$(ts)] S3 done (exit $?)" | tee -a "$LOG"
if [ -f results_scale/S3_v2/metrics.json ]; then
  python3 experiments/diag_routing.py --scale S3 --ckpt results_scale/S3_v2/ckpt_final.pkl >> "$LOG" 2>&1 || echo "[$(ts)] S3 diag failed (non-fatal)" | tee -a "$LOG"
  python3 experiments/report_scale.py >> "$LOG" 2>&1 || echo "[$(ts)] report failed" | tee -a "$LOG"
else
  echo "[$(ts)] WARN: S3 metrics.json missing" | tee -a "$LOG"
fi

echo "[$(ts)] chain complete" | tee -a "$LOG"
