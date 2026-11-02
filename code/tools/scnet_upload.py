"""scnet_upload.py —— 把本地 deploy 包上传到 scnet efile 家目录。

用法：
    python3 scnet_upload.py                 # 上传 /workspace/deploy -> $SCNET_WORK_DIR
    python3 scnet_upload.py --base /workspace/deploy --remote $SCNET_WORK_DIR

依赖 scnet_client.get_tokens() 取 token 与 efile 基址。
"""
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import requests, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scnet_client import get_tokens

REGION = os.environ.get("SCNET_REGION", "")
REMOTE_BASE = os.environ.get("SCNET_WORK_DIR", "")


def efile_base(token):
    req = urllib.request.Request("https://www.scnet.cn/ac/openapi/v2/center",
                                 headers={"token": token})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    return d["data"]["efileUrls"][0]["url"]


def mkdir(base, full_path, token):
    url = base + "/openapi/v2/file/mkdir"
    params = urllib.parse.urlencode({"path": full_path, "createParents": "true"})
    r = requests.post(f"{url}?{params}", headers={"token": token,
                     "Content-Type": "application/json"}, timeout=30)
    return r.status_code, r.text


def upload_file(base, local_path, remote_dir, token):
    with open(local_path, "rb") as f:
        r = requests.post(base + "/openapi/v2/file/upload",
            headers={"token": token},
            files={"file": (os.path.basename(local_path), f)},
            data={"path": remote_dir, "cover": "cover"}, timeout=600)
    return r.status_code, r.text


def main():
    import urllib.parse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=_ROOT)
    ap.add_argument("--remote", default=REMOTE_BASE)
    args = ap.parse_args()

    toks = get_tokens()
    token = [d["token"] for d in toks["data"] if d["clusterId"] == REGION][0]
    base = efile_base(token)
    print(f"[upload] efile={base}\n[upload] remote={args.remote}", flush=True)

    # 1) 建根目录
    sc, tx = mkdir(base, args.remote, token)
    print(f"[mkdir] root -> {sc} {tx[:80]}", flush=True)

    # 2) 遍历本地 deploy，建目录 + 上传文件
    n_files, n_bytes = 0, 0
    for root, dirs, files in os.walk(args.base):
        rel = os.path.relpath(root, args.base)
        remote_dir = args.remote if rel == "." else os.path.join(args.remote, rel)
        # 远程建该目录
        mkdir(base, remote_dir, token)
        for fn in files:
            lp = os.path.join(root, fn)
            sz = os.path.getsize(lp)
            sc, tx = upload_file(base, lp, remote_dir, token)
            n_files += 1; n_bytes += sz
            if sc != 200 or '"code":"0"' not in tx:
                print(f"  [WARN] {remote_dir}/{fn} -> {sc} {tx[:120]}", flush=True)
            else:
                print(f"  ok {remote_dir}/{fn} ({sz/1e6:.1f}MB)  [{n_files} files, {n_bytes/1e6:.0f}MB]", flush=True)
    print(f"[upload] DONE: {n_files} files, {n_bytes/1e6:.0f}MB total", flush=True)


if __name__ == "__main__":
    main()
