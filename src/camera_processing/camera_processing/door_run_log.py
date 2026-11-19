"""Incremental, self-contained record of one door navigation run."""

import base64
from datetime import datetime
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile


class DoorRunLog:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime('door_%Y-%m-%d_%H-%M-%S_')
        self.file = NamedTemporaryFile(dir=directory, prefix=prefix, suffix='.html',
                                       mode='w', encoding='utf-8', delete=False)
        self.path = Path(self.file.name)
        self.file.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                        '<title>四门导航日志</title><body><h1>四门导航日志</h1>\n')
        self.file.flush()

    def event(self, message, jpeg=None):
        timestamp = datetime.now().astimezone().isoformat(timespec='milliseconds')
        self.file.write(f'<p><time>{timestamp}</time> {escape(message)}</p>\n')
        if jpeg is not None:
            data = base64.b64encode(jpeg).decode('ascii')
            self.file.write(f'<img alt="本次发给 Qwen 的相机帧" src="data:image/jpeg;base64,{data}">\n')
        self.file.flush()

    def close(self):
        self.file.close()
