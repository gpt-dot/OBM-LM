"""tokenizer.py —— BPE 分词器（tokenizers 库，ByteLevel，无 UNK）。

为什么 ByteLevel BPE：中英混合 + 代码域。字节级基元保证任何文本可编码
（中文不在词表也有字节兜底），GPT-2 同款预分词/解码；词表 16K 在
"中文覆盖率 × 序列长度 × 嵌入参数量"之间取平衡（实测 8K 时中文频繁
碎成 2-3 字节 token，序列有效信息密度下降 ~30%）。
"""
import os
import numpy as np


class BPETokenizer:
    def __init__(self, tok):
        self.tok = tok

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    @classmethod
    def train(cls, text_samples, vocab_size=16384, min_frequency=2):
        """text_samples: 可迭代的 str（每域取几 MB 混样）。"""
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(text_samples, trainer=trainer)
        return cls(tok)

    def encode(self, text):
        return np.asarray(self.tok.encode(text).ids, dtype=np.int64)

    def encode_batch(self, texts):
        out = []
        for ids in self.tok.encode_batch(texts):
            out.append(np.asarray(ids.ids, dtype=np.int64))
        return out

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tok.save(path)

    @classmethod
    def load(cls, path):
        try:
            from tokenizers import Tokenizer
            return cls(Tokenizer.from_file(path))
        except Exception:
            # 老版 tokenizers 库（如 scnet py37 模块自带的 0.10.x）读不了
            # 新版 tokenizer.json —— 回退纯 Python BPE，接口完全兼容
            from obm_lm.pure_bpe import PureBPETokenizer
            return PureBPETokenizer.load(path)
