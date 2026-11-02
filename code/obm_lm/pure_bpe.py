"""pure_bpe.py —— 纯 Python ByteLevel BPE（无 tokenizers 库依赖）。

为什么：scnet 计算节点的 PyTorch 模块自带老版 HF tokenizers（0.10.x），
读不了新版（0.20+）tokenizer.json（报 "untagged enum ModelWrapper"）。
而训练语料是预编码 npy，tokenizer 只用于 demo/eval 的编解码——
纯 Python 实现足够（速度无所谓，正确性第一）。

实现：GPT-2 同款 byte-level BPE。vocab + merges 从 tokenizer.json 提取。
python3.7 兼容（scnet 节点约束）。
"""
import json
import re


def bytes_to_unicode():
    """GPT-2 标准字节->unicode 映射（可打印区直通，其余平移）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# GPT-2 pretokenize 的近似（无 regex 库的 \p{L}，用 unicode 语义类等价）：
#   [^\W\d_] ≈ \p{L}（unicode 字母）；\d ≈ \p{N}；
#   [^\s\p{L}\p{N}]（标点类，含下划线 _）≈ (?:[^\s\w]|_)
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class PureBPE:
    def __init__(self, vocab, merges, endoftext_id=None):
        """vocab: {token_str: id}；merges: [(a, b), ...]（按 rank 升序）。"""
        self.vocab = dict(vocab)
        self.id2tok = {i: t for t, i in self.vocab.items()}
        self.ranks = {}
        for r, (a, b) in enumerate(merges):
            self.ranks[(a, b)] = r
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.endoftext_id = endoftext_id

    # ---------- 构造 ----------
    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        model = d.get("model", d)
        vocab = model["vocab"]
        raw_merges = model["merges"]
        merges = []
        for m in raw_merges:
            if isinstance(m, str):
                parts = m.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                merges.append((m[0], m[1]))
        eot = vocab.get("<|endoftext|>")
        return cls(vocab, merges, endoftext_id=eot)

    # ---------- 核心 BPE ----------
    def _bpe(self, symbols):
        """symbols: list[str]（byte-unicode 字符串）-> 合并后的 token 串列表。"""
        while len(symbols) > 1:
            best = None
            best_rank = None
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = i
            if best is None:
                break
            symbols = (symbols[:best] + [symbols[best] + symbols[best + 1]]
                       + symbols[best + 2:])
        return symbols

    def encode(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        ids = []
        for word in _PAT.findall(text):
            unit = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
            toks = self._bpe(list(unit))
            for t in toks:
                i = self.vocab.get(t)
                if i is None:  # 理论不发生（byte alphabet 全在词表）
                    for ch in t:
                        i2 = self.vocab.get(ch)
                        if i2 is not None:
                            ids.append(i2)
                else:
                    ids.append(i)
        return ids

    def decode(self, ids):
        buf = []
        for i in ids:
            t = self.id2tok.get(int(i), "")
            if t == "<|endoftext|>":
                continue
            buf.append(t)
        s = "".join(buf)
        raw = bytearray()
        for ch in s:
            b = self.byte_decoder.get(ch)
            if b is not None:
                raw.append(b)
        return raw.decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return len(self.vocab)


class PureBPETokenizer:
    """与 BPETokenizer 同接口（encode/decode/vocab_size）。"""

    def __init__(self, bpe):
        self.tok = bpe

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    @classmethod
    def load(cls, path):
        return cls(PureBPE.from_json(path))

    def encode(self, text):
        import numpy as np
        return np.asarray(self.tok.encode(text), dtype=np.int64)

    def decode(self, ids):
        return self.tok.decode(ids)
