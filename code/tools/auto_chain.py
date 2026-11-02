"""auto_chain.py —— 队列套利自动链（头脑风暴 #1，TRIZ No.10 预先作用）。

轮询 scale7 作业，COMPLETED 立即提交 scale8（权重已在远端节点，无需下载），
把排队/完工的空转时间压到零。FAILED 则只报警不提交。
"""
import os, sys, time, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scnet_submit import region_token, job_detail

JOB_S7 = os.environ.get("SCNET_JOB_ID", "")
CMD = ["python3", "scnet_submit.py", "--script", "deploy/train8.sh",
       "--name", "scale8_balance", "--wall", "06:00:00"]

while True:
    d = job_detail(region_token(), JOB_S7)
    st = d.get("data", {}).get("jobStatus") or "??"
    print(f'[{time.strftime("%H:%M:%S")}] scale7={st}', flush=True)
    if st.lower() in ("statc", "completed", "completing"):
        print("[chain] scale7 done -> submitting scale8", flush=True)
        r = subprocess.run(CMD, cwd=BASE, capture_output=True, text=True)
        print("[chain] submit rc=", r.returncode, flush=True)
        print(r.stdout[-500:], r.stderr[-300:], flush=True)
        break
    if st.lower() in ("failed", "cancelled", "timeout", "node_fail"):
        print(f"[chain] scale7 {st} -> NOT submitting, need human", flush=True)
        break
    time.sleep(120)
