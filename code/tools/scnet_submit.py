"""scnet_submit.py —— 提交作业到 scnet 并轮询状态。

用法：
    python3 scnet_submit.py --script deploy/smoke.sh --name smoke --wall 00:10:00
    python3 scnet_submit.py --script deploy/train.sh --name scale2 --wall 23:00:00 --poll
"""
import os, sys, time, json, argparse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scnet_client import get_tokens, submit_job, api_get

REGION = os.environ.get("SCNET_REGION", "")
JOBMANAGER_ID = os.environ.get("SCNET_JOBMANAGER_ID", "")
HPC_BASE = os.environ.get("SCNET_HPC_BASE", "")
QUEUE = os.environ.get("SCNET_QUEUE", "Agent0")
WORK_DIR = os.environ.get("SCNET_WORK_DIR", "")


def region_token():
    toks = get_tokens()
    return [d["token"] for d in toks["data"] if d["clusterId"] == REGION][0]


def job_detail(token, job_id):
    try:
        return api_get(HPC_BASE, f"/hpc/openapi/v2/jobs/{job_id}", token)
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--name", default="obm_job")
    ap.add_argument("--wall", default="01:00:00")
    ap.add_argument("--ppn", type=int, default=32)
    ap.add_argument("--poll", action="store_true", help="提交后轮询直到结束")
    args = ap.parse_args()

    token = region_token()
    with open(args.script) as f:
        cmd = f.read()
    code, msg, job_id = submit_job(token, HPC_BASE, JOBMANAGER_ID, REGION,
                                   cmd, WORK_DIR, QUEUE, args.wall,
                                   ppn=args.ppn, name=args.name)
    print(f"[submit] code={code} msg={msg} job_id={job_id} name={args.name}")
    if str(code) != "0" or not job_id:
        print("[submit] FAILED"); sys.exit(1)

    if args.poll:
        while True:
            time.sleep(30)
            d = job_detail(token, job_id)
            st = d.get("data", {})
            state = st.get("jobStatus") or st.get("state") or st.get("status") or d.get("msg")
            out = st.get("outputPath", "")
            print(f"  [{time.strftime('%H:%M:%S')}] job {job_id} status={state}", flush=True)
            if isinstance(state, str) and state.upper() in ("COMPLETED", "COMPLETING", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                print("[submit] job finished:", state, "| output:", out)
                break
    else:
        print(f"[submit] job {job_id} submitted; poll via job_detail.")


if __name__ == "__main__":
    main()
