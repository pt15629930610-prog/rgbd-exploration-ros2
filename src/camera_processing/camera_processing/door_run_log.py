"""Incremental, self-contained record of one door navigation run."""

import base64
from datetime import datetime
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile

from .door_log_sync import HTML_HEADER, LogMirror


RUN_LOG_DIRECTORY = Path.home() / 'door_nav_logs'


class DoorRunLog:
    def __init__(self, directory, mirror_url=None):
        directory.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime('door_%Y-%m-%d_%H-%M-%S_')
        self.file = NamedTemporaryFile(dir=directory, prefix=prefix, suffix='.html',
                                       mode='w', encoding='utf-8', newline='\n', delete=False)
        self.path = Path(self.file.name)
        self.file.write(HTML_HEADER)
        self.file.flush()
        self.mirror = LogMirror(self.path, mirror_url) if mirror_url else None

    def event(self, message, jpeg=None):
        timestamp = datetime.now().astimezone().isoformat(timespec='milliseconds')
        safe = escape(message).replace('\r', '&#13;').replace('\n', '&#10;')
        self.file.write(f'<p><time>{timestamp}</time> {safe}</p>\n')
        if jpeg is not None:
            data = base64.b64encode(jpeg).decode('ascii')
            self.file.write(f'<img alt="本次门观测的 RGB 图" src="data:image/jpeg;base64,{data}">\n')
        self.file.flush()

    def close(self):
        self.file.close()
        if self.mirror is not None:
            self.mirror.close()
