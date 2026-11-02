"""night_pipeline.py —— 无人值守夜间流水线（头脑风暴 #1，No.10 预先作用）。

盯远端训练作业（作业 ID 由 SCNET_JOB_ID 指定）：
  statC -> 自动下载权重 + 成绩单 -> 自动生成跨代战报 HTML
不再自动提交 scale9（其配置依赖语料 encode 完成，需人工确认）。
"""
import sys, time, os, json, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scnet_client import get_tokens, efile_base, download_file
from scnet_submit import region_token, job_detail

JOB = os.environ.get("SCNET_JOB_ID", "")
REGION = os.environ.get("SCNET_REGION", "")
REMOTE_HOME = os.environ.get("REMOTE_HOME", "/public/home/your_account")
FILES = [("results_lab/scale8.log", os.path.join(BASE, "_s8.log")),
         ("results_lab/scale8.json", os.path.join(BASE, "_s8.json")),
         ("results_lab/scale8_weights.pkl",
          os.path.join(BASE, "OBM-LM-scale8-训后-75M.pkl"))]


def dl_all():
    toks = get_tokens()
    token = [d["token"] for d in toks["data"] if d["clusterId"] == REGION][0]
    base = efile_base(token)
    for rp, lp in FILES:
        for att in range(3):
            try:
                download_file(token, base,
                              f"{REMOTE_HOME}/obm_mvp/{rp}", lp)
                print(f"[night] ok {rp}", flush=True)
                break
            except Exception as e:
                print(f"[night] retry {rp}: {str(e)[:60]}", flush=True)
                time.sleep(10)


def report():
    """跨代战报：scale4-8 指标表 + 事件 + 看门狗日志。"""
    try:
        j8 = json.load(open(os.path.join(BASE, "_s8.json")))
    except Exception:
        j8 = {}
    hist = {"scale4": dict(en_lit=601, zh_classic=963, code_py=1327, en_wiki=4596),
            "scale5": dict(en_lit=163, zh_classic=1557, code_py=291, en_wiki=295),
            "scale6": dict(en_lit=123, zh_classic=825, code_py=284, en_wiki=391),
            "scale7": dict(en_lit=120, zh_classic=837, code_py=73, en_wiki=296)}
    fp = j8.get("final_ppl") or {}
    if fp:
        hist["scale8"] = {k: round(v) for k, v in fp.items()}
    doms = ["en_lit", "zh_classic", "code_py", "en_wiki"]
    rows = ""
    for d in doms:
        cells = "".join(
            f"<td>{hist[s].get(d, '-')}</td>"
            for s in ("scale4", "scale5", "scale6", "scale7", "scale8"))
        best = min((hist[s][d], s) for s in hist if d in hist[s])
        rows += f"<tr><td>{d}</td>{cells}<td class='ok'>{best[0]}（{best[1]}）</td></tr>"
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>OBM-LM scale8 跨代战报</title><style>
body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f6f8fa;margin:0}}
.wrap{{max-width:900px;margin:0 auto;background:#fff;padding:30px 36px 60px}}
h1{{font-size:22px;border-bottom:3px solid #1a5fb4;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0}}
th,td{{border:1px solid #dfe6ee;padding:7px 10px;text-align:center}}
th{{background:#eef2f7}} .ok{{color:#26a269;font-weight:600}}
</style></head><body><div class="wrap">
<h1>OBM-LM scale8 跨代战报（自动生成 @ {time.strftime('%m-%d %H:%M')}）</h1>
<table><tr><th>域</th><th>scale4</th><th>scale5</th><th>scale6</th><th>scale7</th>
<th>scale8</th><th>历史最佳</th></tr>{rows}</table>
<pre style="background:#fafbfd;border:1px solid #e6ebf0;padding:12px;font-size:12.5px">{open(os.path.join(BASE, '_s8.log')).read()[-3000:] if os.path.exists(os.path.join(BASE, '_s8.log')) else ''}</pre>
</div></body></html>"""
    open(os.path.join(BASE, "OBM-LM-scale8跨代战报.html"), "w").write(html)
    print("[night] report written", flush=True)


while True:
    d = job_detail(region_token(), JOB)
    st = d.get("data", {}).get("jobStatus") or "??"
    print(f'[{time.strftime("%H:%M:%S")}] scale8={st}', flush=True)
    if st.lower() in ("statc", "completed", "completing"):
        dl_all()
        report()
        # No.10 预先作用接力：scale8 完工 -> 若语料已上传即自动提交 scale9
        if os.path.exists(os.path.join(BASE, ".corpus_uploaded")):
            for att in range(3):
                try:
                    r = subprocess.run(
                        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scnet_submit.py"),
                         "--script", os.path.join(BASE, "train9.sh"),
                         "--name", "scale9_data", "--wall", "08:00:00"],
                        capture_output=True, text=True, timeout=120)
                    print(f"[night] scale9 submit: "
                          f"{(r.stdout or r.stderr)[-200:]}", flush=True)
                    if r.returncode == 0:
                        break
                except Exception as e:
                    print(f"[night] submit retry: {str(e)[:80]}", flush=True)
                    time.sleep(30)
        else:
            print("[night] corpus not yet uploaded -> scale9 不自动提交，"
                  "待语料就绪后人工或 auto_chain 触发", flush=True)
        print("[night] pipeline done", flush=True)
        break
    if st.lower() in ("failed", "cancelled", "timeout", "node_fail"):
        print(f"[night] scale8 {st} -> alert, no auto action", flush=True)
        break
    time.sleep(180)
