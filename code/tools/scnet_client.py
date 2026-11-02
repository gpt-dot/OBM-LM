"""SCNet HPC OpenAPI 客户端（AK/SK 签名）。仅含只读探测与作业提交能力。

凭据一律通过环境变量提供，不写入代码：
  SCNET_USER / SCNET_AK / SCNET_SK
"""
import hmac, hashlib, time, json, os, urllib.request, urllib.error, urllib.parse

USER = os.environ.get("SCNET_USER", "")
AK = os.environ.get("SCNET_AK", "")
SK = os.environ.get("SCNET_SK", "")

def sign(ts=None):
    if not (USER and AK and SK):
        raise RuntimeError("请先设置 SCNET_USER / SCNET_AK / SCNET_SK 环境变量")
    ts = ts or str(int(time.time()))
    body = '{"accessKey":"%s","timestamp":"%s","user":"%s"}' % (AK, ts, USER)
    sig = hmac.new(SK.encode(), body.encode(), hashlib.sha256).hexdigest().lower()
    return ts, body, sig

def get_tokens():
    ts, body, sig = sign()
    req = urllib.request.Request(
        "https://api.scnet.cn/api/user/v3/tokens",
        data=b"{}", method="POST",
        headers={"Content-Type": "application/json", "user": USER,
                 "accessKey": AK, "timestamp": ts, "signature": sig})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def region_url(token):
    """GET 获取某区域的授权 URL（hpcUrls/efileUrls 等）。"""
    url = "https://api.scnet.cn/ac/openapi/v2/center"
    req = urllib.request.Request(url, headers={"token": token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def api_get(base, path, token, params=None):
    q = ("?" + "&".join(f"{k}={v}" for k, v in (params or {}).items())) if params else ""
    req = urllib.request.Request(base.rstrip("/") + path + q,
        headers={"token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def upload_file(local_path, remote_dir, token, efile_base):
    """上传单个文件到远端目录（token 鉴权，无需 Cookie）。"""
    import requests
    with open(local_path, "rb") as f:
        r = requests.post(efile_base + "/openapi/v2/file/upload",
            headers={"token": token},
            files={"file": (os.path.basename(local_path), f)},
            data={"path": remote_dir, "cover": "cover"}, timeout=600)
    return r.status_code, r.text


def submit_job(token, hpc_base, jobmanager_id, cluster_id, cmd_content,
               work_dir, queue, walltime, ppn=32, nnode=1, name="obm_job"):
    """提交 SLURM 作业（BASIC/BASE 模板，cmd 模式：脚本内容放 GAP_CMD_FILE）。

    返回 (code, msg, job_id)。walltime 用 HH:MM:ss。
    """
    url = f"{hpc_base}/hpc/openapi/v2/apptemplates/BASIC/BASE/job"
    body = {
        "strJobManagerID": str(jobmanager_id),
        "mapAppJobInfo": {
            "GAP_CMD_FILE": cmd_content,
            "GAP_NNODE": str(nnode),
            "GAP_NODE_STRING": "",
            "GAP_SUBMIT_TYPE": "cmd",
            "GAP_JOB_NAME": name,
            "GAP_WORK_DIR": work_dir,
            "GAP_QUEUE": queue,
            "GAP_NPROC": "",
            "GAP_PPN": str(ppn),
            "GAP_NGPU": "",
            "GAP_NDCU": "",
            "GAP_WALL_TIME": walltime,
            "GAP_EXCLUSIVE": "",
            "GAP_APPNAME": "BASE",
            "GAP_MULTI_SUB": "",
            "GAP_STD_OUT_FILE": os.path.join(work_dir, "std.out.%j"),
            "GAP_STD_ERR_FILE": os.path.join(work_dir, "std.err.%j"),
            "GAP_SCHEDULER_OPT_WEB": "",
        },
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        method="POST", headers={"token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return d.get("code"), d.get("msg"), d.get("data")


def job_detail(token, hpc_base, job_id):
    """查询作业详情（含状态/输出路径）。"""
    return api_get(f"{hpc_base}/hpc/openapi/v2/jobs/{job_id}", token)


def download_file(token, efile_base, remote_path, local_path):
    """从 efile 下载单个文件到本地。返回 (code, local_path)。"""
    import requests, os
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    url = efile_base + "/openapi/v2/file/download"
    params = urllib.parse.urlencode({"path": remote_path})
    r = requests.get(f"{url}?{params}", headers={"token": token}, timeout=600, stream=True)
    if r.status_code != 200:
        return r.status_code, r.text[:200]
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    return 0, local_path


def efile_base(token):
    req = urllib.request.Request("https://www.scnet.cn/ac/openapi/v2/center",
                                 headers={"token": token})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    return d["data"]["efileUrls"][0]["url"]

if __name__ == "__main__":
    toks = get_tokens()
    print("=== tokens ===")
    regions = {}
    for d in toks["data"]:
        if d["clusterId"] == "0":
            continue
        print(f"  {d['clusterName']} (id={d['clusterId']}) status=Normal")
        regions[d["clusterId"]] = d["token"]
    # 取第一个区域拿 URL
    first_id = next(iter(regions))
    centers = region_url(regions[first_id])
    print("=== center (华东一区) ===")
    print(json.dumps(centers, ensure_ascii=False)[:1200])
