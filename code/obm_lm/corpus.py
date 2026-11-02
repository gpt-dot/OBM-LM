"""corpus.py —— 多领域真实文本语料 → OBM 域流协议。

四个领域（token 分布差异大，签名空间可分离）：
    en_lit       英文文学（古腾堡小说 + 莎士比亚）
    zh_classic   中文古典（西游记/金瓶梅/呐喊等，古腾堡中文书）
    code_py      Python 源码（GitHub 仓库 tar.gz）
    en_wiki      英文维基（wikitext-103，含 == 标题 ==/[[链接]] 标记）

6 段协议（与玩具实验 langs.build_blocks 同构）：
    [en_lit, zh_classic, code_py, en_wiki, en_lit*, zh_classic*]
    后两段是复现——测抗遗忘与复现恢复。每域尾部留出作固定评估窗口。

域内 batch：从该域 train token 流里随机截 ctx+1 长的窗口，seq=window[:-1]、
tgt=window[1:]（标准 next-token LM；与 mesh 的 (seq,tgt,mask) 接口对齐，
mask 为全 True 的逐位置 mask——真实 LM 没有玩具任务的"输入段"）。
"""
import os
import re
import json
import tarfile
import numpy as np

# 根目录可经环境变量 OBM_ROOT 覆盖（默认本地沙箱路径），便于远端部署
_OBM_ROOT = os.environ.get("OBM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(_OBM_ROOT, "data", "raw")
CACHE = os.path.join(_OBM_ROOT, "data", "cache")

DOMAINS = ["en_lit", "zh_classic", "code_py", "en_wiki"]
PLAN = ["en_lit", "zh_classic", "code_py", "en_wiki", "en_lit", "zh_classic"]


# ================= 原始文件 -> 纯文本 =================
_GUT_HEAD = re.compile(r"\*\*\* ?START OF (TH|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       re.S)
_GUT_TAIL = re.compile(r"\*\*\* ?END OF (TH|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       re.S)


def _read_text(path):
    """读文本文件，猜编码（中文古腾堡书有 UTF-8/GBK 两种）。"""
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gb18030", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _strip_gutenberg(text):
    m = _GUT_HEAD.search(text)
    if m:
        text = text[m.end():]
    m = _GUT_TAIL.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def load_en_lit(raw=RAW):
    out = []
    d = os.path.join(raw, "en_lit")
    for fn in sorted(os.listdir(d)):
        t = _strip_gutenberg(_read_text(os.path.join(d, fn)))
        if len(t) > 5000:
            out.append(t)
    return "\n\n".join(out)


def load_zh_classic(raw=RAW):
    out = []
    d = os.path.join(raw, "zh_classic")
    for fn in sorted(os.listdir(d)):
        t = _strip_gutenberg(_read_text(os.path.join(d, fn)))
        if len(t) > 5000:
            out.append(t)
    return "\n\n".join(out)


def load_code_py(raw=RAW, max_files=12000):
    out = []
    d = os.path.join(raw, "code_py")
    tars = sorted(fn for fn in os.listdir(d) if fn.endswith(".tar.gz"))
    per_tar = max(1, max_files // max(1, len(tars)))
    for fn in tars:
        try:
            with tarfile.open(os.path.join(d, fn)) as tf:
                n = 0
                for m in tf.getmembers():
                    if not m.name.endswith(".py") or m.size < 200 or m.size > 200000:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        t = f.read().decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    if len(t) > 200:
                        out.append(f"# ===== {fn}:{m.name} =====\n{t}")
                        n += 1
                    if n >= per_tar:
                        break
        except tarfile.TarError:
            continue
    return "\n\n".join(out)


def load_en_wiki(raw=RAW, max_chars=120_000_000):
    """wikitext-103 parquet 分片(可多个) -> 纯文本（保留 == 标题 ==/[[链接]] 标记——
    标记本身就是该域区别于 en_lit 的签名特征）。"""
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ""
    paths = sorted(glob.glob(os.path.join(raw, "en_wiki", "*.parquet")))
    out, total = [], 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            for rec in batch.to_pylist():
                t = rec["text"]
                if not t or not t.strip():
                    continue
                out.append(t)
                total += len(t)
                if total >= max_chars:
                    return "\n".join(out)
    return "\n".join(out)


LOADERS = {"en_lit": load_en_lit, "zh_classic": load_zh_classic,
           "code_py": load_code_py, "en_wiki": load_en_wiki}


# ================= 分词缓存 =================
def build_token_cache(tok, cache=CACHE, eval_tokens=120_000, force=False):
    """每域：全文本 -> token ids（int32 存储，省一半磁盘/上传流量），
    尾部留 eval_tokens 作评估，存 .npy。"""
    os.makedirs(cache, exist_ok=True)
    info = {}
    for name, loader in LOADERS.items():
        dst = os.path.join(cache, f"{name}_tokens.npy")
        if os.path.exists(dst) and not force:
            info[name] = dict(n=int(np.load(dst, mmap_mode="r").shape[0]))
            continue
        text = loader()
        ids = np.asarray(tok.encode(text), dtype=np.int32)
        np.save(dst, ids)
        info[name] = dict(n=int(ids.shape[0]), chars=len(text))
        print(f"  [cache] {name}: {len(text)/1e6:.1f}MB text -> "
              f"{ids.shape[0]/1e6:.2f}M tokens", flush=True)
    return info


class DomainStream:
    """一个概念段：train token 流 + 固定评估窗口。batch 接口与玩具
    LangBlock 同构：(seq, tgt, mask)，seq/tgt 为 (B, ctx) 的 next-token 对。"""

    def __init__(self, name, tokens, seed, eval_tokens=120_000):
        self.name = name
        n = tokens.shape[0]
        self.tokens = tokens[:n - eval_tokens] if n > eval_tokens * 2 else tokens
        self.eval_tokens = tokens[n - eval_tokens:] if n > eval_tokens * 2 else tokens
        self.rng = np.random.default_rng(seed)

    def batch(self, bs, ctx):
        hi = self.tokens.shape[0] - ctx - 1
        starts = self.rng.integers(0, hi, size=bs)
        seq = np.stack([self.tokens[s:s + ctx] for s in starts]).astype(np.int64)
        tgt = np.stack([self.tokens[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
        return seq, tgt, np.ones(ctx, dtype=bool)

    def eval_windows(self, n=24, ctx=256):
        """固定评估窗口（去重 seed，可复现）。"""
        rng = np.random.default_rng(9999)
        hi = self.eval_tokens.shape[0] - ctx - 1
        if hi <= 0:
            hi = max(1, self.eval_tokens.shape[0] - ctx - 1)
        starts = rng.integers(0, hi, size=n)
        seq = np.stack([self.eval_tokens[s:s + ctx] for s in starts]).astype(np.int64)
        tgt = np.stack([self.eval_tokens[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
        return seq, tgt


def build_streams(cache=CACHE):
    """返回 {domain: DomainStream}；种子与玩具协议同风格（每段独立）。"""
    streams = {}
    for i, name in enumerate(PLAN):
        key = name if i < len(DOMAINS) else f"{name}#repris{i}"
        toks = np.load(os.path.join(cache, f"{name}_tokens.npy"))
        streams[key] = DomainStream(name, toks, seed=1000 + i)
    return streams


def train_tokenizer(sample_mb_per_domain=6, vocab_size=16384, cache=CACHE):
    """在四域混样上训练 BPE（每域取前 sample_mb 字符）。"""
    from obm_lm.tokenizer import BPETokenizer
    texts = []
    for name, loader in LOADERS.items():
        t = loader()
        texts.append(t[:sample_mb_per_domain * 1_000_000])
        print(f"  [bpe sample] {name}: {len(texts[-1])/1e6:.1f}MB", flush=True)
    tok = BPETokenizer.train(texts, vocab_size=vocab_size)
    os.makedirs(cache, exist_ok=True)
    tok.save(os.path.join(cache, "tokenizer.json"))
    return tok
