"""上传新语料缓存到 scnet（带重试续传，后台跑）。"""
import os, sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scnet_upload import efile_base, mkdir, upload_file
from scnet_client import get_tokens

REGION = os.environ.get('SCNET_REGION', '')
HOME = os.environ.get('REMOTE_HOME', '/public/home/your_account')
FILES = [
    (os.path.join(_ROOT, 'obm_lm/corpus.py'),       f'{HOME}/obm_mvp/obm_lm'),
    (os.path.join(_ROOT, 'data/cache/en_lit_tokens.npy'),   f'{HOME}/obm_mvp/data/cache'),
    (os.path.join(_ROOT, 'data/cache/zh_classic_tokens.npy'), f'{HOME}/obm_mvp/data/cache'),
    (os.path.join(_ROOT, 'data/cache/code_py_tokens.npy'),  f'{HOME}/obm_mvp/data/cache'),
    (os.path.join(_ROOT, 'data/cache/en_wiki_tokens.npy'),  f'{HOME}/obm_mvp/data/cache'),
]

toks = get_tokens()
token = [d['token'] for d in toks['data'] if d['clusterId'] == REGION][0]
base = efile_base(token)

for lp, rp in FILES:
    ok = False
    for attempt in range(4):
        try:
            mkdir(base, rp.rsplit('/', 1)[0], token) if False else None
            sc, tx = upload_file(base, lp, rp, token)
            print(f'[{time.strftime("%H:%M")}] {lp.split("/")[-1]} -> {sc}', flush=True)
            if sc == 200:
                ok = True
                break
        except Exception as e:
            print(f'  retry {attempt+1}: {str(e)[:80]}', flush=True)
            time.sleep(20)
    if not ok:
        print(f'FAILED {lp}', flush=True)
print('UPLOAD_DONE', flush=True)
