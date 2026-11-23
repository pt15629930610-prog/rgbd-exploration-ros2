"""Resume the pinned official model download and verify its LFS SHA-256 before use."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
import urllib.request


REPO = 'Qwen/Qwen3-VL-2B-Instruct'
REV = '89644892e4d85e24eaac8bacfd4f463576704203'
FILES = [
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.json',
    'preprocessor_config.json',
    'generation_config.json',
    'video_preprocessor_config.json',
    'vocab.json',
    'merges.txt',
]


def get(url, headers=None):
    for attempt in range(10):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers or {}), timeout=80
            ) as result:
                return result.read()
        except OSError:
            if attempt == 9:
                raise
            time.sleep(min(8, 1 + attempt))


def fetch(folder, source):
    folder.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/'
    weights_url = (
        f'https://modelscope.cn/models/{REPO}/resolve/master/' if source == 'modelscope' else url
    )
    for name in FILES:
        path = folder / name
        if not path.is_file():
            path.write_bytes(get(url + name))
    size = 4255140312
    sha = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
    target = folder / 'model.safetensors'

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as src:
            for block in iter(lambda: src.read(4 << 20), b''):
                h.update(block)
        return h.hexdigest()

    if not (target.exists() and digest(target) == sha):
        chunk = 8 << 20
        parts = [folder / f'model.safetensors.part{i}' for i in range((size + chunk - 1) // chunk)]

        def download(entry):
            i, path = entry
            start = i * chunk
            end = min(size, (i + 1) * chunk) - 1
            if path.exists() and path.stat().st_size == end - start + 1:
                return
            data = get(weights_url + f'model.safetensors?piece={i}', {'Range': f'bytes={start}-{end}'})
            if len(data) != end - start + 1:
                raise ValueError(f'Short response for piece {i}')
            path.write_bytes(data)
            print('downloaded piece', i, flush=True)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(download, enumerate(parts)))
        tmp = target.with_suffix('.assembling')
        with tmp.open('wb') as out:
            for path in parts:
                out.write(path.read_bytes())
        if digest(tmp) != sha:
            raise ValueError('SHA-256 mismatch, refusing to publish weights')
        tmp.replace(target)
        for path in parts:
            path.unlink()
    (folder / 'revision.txt').write_text(REV, encoding='utf8')
    print('Verified', size, 'bytes, SHA-256', sha, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', choices=('huggingface', 'modelscope'), default='huggingface')
    a = p.parse_args()
    fetch(a.root / 'base', a.source)
