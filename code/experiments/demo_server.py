"""demo_server.py —— 细胞动物园交互式演示（头脑风暴 #1）。

Flask 单文件：加载 zoo（多代权重合并）+ v3 自适应路由推理栈，
浏览器输入 prompt -> 返回生成文本 + 细胞路由权重。

用法：python3 experiments/demo_server.py [--port 8765]
      权重用环境变量 OBM_ZOO="pkl1,pkl2" 覆盖，默认 scale5+scale6。
"""
import os, sys, time, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from flask import Flask, request, jsonify

from gen_archive import load_zoo, generate_top1, PROMPTS, _find_model
from obm_lm.tokenizer import BPETokenizer
from obm_lm import corpus, backend as bk

_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_OBM_ROOT, "data", "cache")

app = Flask(__name__)
TOK = None
MESH = None
STREAMS = None
GARDENER = None   # 缓存：逐细胞×逐域 ppl


def get_model():
    global TOK, MESH
    if MESH is None:
        zoo = os.environ.get("OBM_ZOO") or ",".join(
            _find_model(t) for t in ("scale5", "scale6"))
        TOK = BPETokenizer.load(os.path.join(CACHE, "tokenizer.json"))
        t0 = time.time()
        MESH = load_zoo(zoo.split(","), budget_gb=5.5)
        print(f"[server] zoo loaded: {len(MESH.cells)} cells "
              f"({time.time()-t0:.0f}s)", flush=True)
    return MESH


HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>OBM-LM 细胞动物园演示</title>
<style>
body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;background:#f6f8fa;margin:0;
display:flex;justify-content:center;padding:40px 16px}
.card{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.08);
max-width:760px;width:100%;padding:28px 32px}
h1{font-size:20px;margin:0 0 4px;color:#1a5fb4}
.sub{color:#5a6a7e;font-size:13px;margin-bottom:18px}
textarea{width:100%;box-sizing:border-box;border:1px solid #dfe6ee;border-radius:8px;
padding:10px;font-size:14px;font-family:inherit;resize:vertical;min-height:64px}
.row{display:flex;gap:10px;margin:12px 0;flex-wrap:wrap;align-items:center}
.row label{font-size:13px;color:#1a2332}
button{background:#1a5fb4;color:#fff;border:0;border-radius:8px;padding:10px 26px;
font-size:14px;cursor:pointer}
button:disabled{background:#9db8d8}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.chip{background:#eef2f7;border:1px solid #dfe6ee;border-radius:14px;padding:3px 12px;
font-size:12px;cursor:pointer;color:#1a2332}
#out{margin-top:16px}
#route{margin:10px 0;font-size:12.5px;color:#5a6a7e}
.bar{display:inline-block;height:10px;background:#1a5fb4;border-radius:5px;
vertical-align:middle;margin:0 6px 0 4px}
#text{white-space:pre-wrap;background:#fafbfd;border:1px solid #e6ebf0;border-radius:8px;
padding:14px;font-size:14px;line-height:1.7;min-height:60px}
#meta{font-size:12px;color:#8a97a8;margin-top:6px}
</style></head><body><div class="card">
<h1>OBM-LM 细胞动物园演示</h1>
<div class="sub">多代细胞合并 + 前缀 NLL 自适应路由（自信单细胞 / 模糊软混合）· 75M 级 · 8GB 沙箱推理</div>
<div class="chips" id="chips"></div>
<textarea id="prompt" placeholder="输入 prompt（中英文/代码均可，路由会自动选细胞）"></textarea>
<div class="row">
<label>温度 <span id="tv">0.80</span> <input type="range" id="temp" min="50" max="120" value="80" style="vertical-align:middle"></label>
<label>生成长度 <input type="number" id="genlen" value="48" min="8" max="128" style="width:56px"></label>
<button id="go">生成</button><span id="status"></span>
</div>
<div id="out" style="display:none">
<div id="route"></div>
<div id="text"></div>
<div id="meta"></div>
</div>
</div>
<script>
const prompts = %PROMPTS%;
const chips = document.getElementById('chips');
prompts.forEach(([dom, p]) => {
  const s = document.createElement('span');
  s.className = 'chip'; s.textContent = dom + '｜' + p.slice(0, 18);
  s.onclick = () => { document.getElementById('prompt').value = p; };
  chips.appendChild(s);
});
const temp = document.getElementById('temp');
temp.oninput = () => document.getElementById('tv').textContent =
  (temp.value / 100).toFixed(2);
document.getElementById('go').onclick = async () => {
  const btn = document.getElementById('go');
  btn.disabled = true; document.getElementById('status').textContent = '生成中…';
  try {
    const r = await fetch('/api/gen', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: document.getElementById('prompt').value,
        temp: temp.value / 100,
        gen_len: parseInt(document.getElementById('genlen').value),
        seed: Math.floor(Math.random() * 1e6)})});
    const d = await r.json();
    if (d.error) { document.getElementById('status').textContent = '出错: ' + d.error; return; }
    document.getElementById('out').style.display = 'block';
    let html = '路由：';
    const total = Object.values(d.route).reduce((a, b) => a + b, 0);
    for (const [cid, w] of Object.entries(d.route)) {
      const pct = Math.round(100 * w / total);
      html += `细胞#${cid} <span class="bar" style="width:${pct * 2}px"></span> ${pct}%&nbsp;&nbsp;`;
    }
    document.getElementById('route').innerHTML = html;
    document.getElementById('text').textContent = d.text;
    document.getElementById('meta').textContent =
      `${d.sec.toFixed(1)}s · ${d.n_cells} 细胞池 · ${d.mode}`;
    document.getElementById('status').textContent = '';
  } finally { btn.disabled = false; }
};
</script></body></html>"""


@app.route("/")
def index():
    return HTML.replace("%PROMPTS%", str(PROMPTS))


@app.route("/api/gen", methods=["POST"])
def api_gen():
    d = request.get_json(force=True)
    prompt = (d.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "empty prompt"})
    mesh = get_model()
    temp = d.get("temp")
    text, sec, route = generate_top1(
        mesh, TOK, prompt,
        gen_len=int(d.get("gen_len", 48)),
        temp=(float(temp) if temp else None),
        top_k=40, seed=int(d.get("seed", 0)))
    return jsonify({"text": prompt + text, "route": route,
                    "sec": sec, "n_cells": len(mesh.cells),
                    "mode": "adaptive-NLL（zoo）"})


def gardener_data():
    """细胞园丁（头脑风暴 #3）：逐细胞×逐域 ppl 饲养状态（首次请求算好缓存）。"""
    global GARDENER
    if GARDENER is None:
        global STREAMS
        mesh = get_model()
        if STREAMS is None:
            STREAMS = corpus.build_streams(CACHE)
        data = {}
        for k in corpus.DOMAINS:
            seq, tgt = STREAMS[k].eval_windows(ctx=mesh.cfg.max_len)
            seq, tgt = seq[:24], tgt[:24]
            row = []
            for c in mesh.cells:
                vals = [bk.mean_nll(c, seq[b:b+12], tgt[b:b+12])
                        for b in range(0, len(seq), 12)]
                row.append(round(float(np.exp(np.mean(vals)))))
            data[k] = row
        GARDENER = {"cell_ids": [c.cell_id for c in mesh.cells], "ppl": data}
    return GARDENER


GARDENER_HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>OBM-LM 细胞园丁</title><style>
body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;background:#f6f8fa;
display:flex;justify-content:center;padding:40px 16px;margin:0}
.card{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.08);
max-width:720px;width:100%;padding:28px 32px}
h1{font-size:20px;color:#1a5fb4;margin:0 0 4px}
.sub{color:#5a6a7e;font-size:13px;margin-bottom:16px}
table{border-collapse:collapse;width:100%%;font-size:13px}
th,td{border:1px solid #dfe6ee;padding:7px 10px;text-align:left}
th{background:#eef2f7}
.bar{display:inline-block;height:12px;background:#1a5fb4;border-radius:6px;vertical-align:middle}
.legend{font-size:12px;color:#5a6a7e;margin-top:10px}
a{color:#1a5fb4;font-size:13px}
</style></head><body><div class="card">
<h1>OBM-LM 细胞园丁 🌱</h1>
<div class="sub">逐细胞 × 逐域困惑度——每个细胞的"饲养状态"（越短越强）</div>
<table><tr><th>域</th>%ROWS%</table>
<div class="legend">细胞编号：%CELLS% · 条形越长 ppl 越大（越弱）。列表数字为 ppl。</div>
<p><a href="/">← 返回生成演示</a></p>
</div></body></html>"""


@app.route("/gardener")
def gardener():
    d = gardener_data()
    ids = d["cell_ids"]
    rows = ""
    for k, ppls in d["ppl"].items():
        mx = max(ppls)
        cells = "".join(
            f'<td><span class="bar" style="width:{min(200, p/mx*160)}px"></span> '
            f'{p}</td>' for p in ppls)
        rows += f'<tr><td>{k}</td>{cells}</tr>'
    head = "".join(f"<th>细胞#{c}</th>" for c in ids)
    html = GARDENER_HTML.replace("%ROWS%", f"<th></th>{head}" + rows)
    return html.replace("%CELLS%", "、".join(f"#{c}" for c in ids))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    get_model()  # 启动即加载，避免首个请求超时
    app.run(host="0.0.0.0", port=args.port, threaded=False)
