"""A shareable run record is readable before the simulator stops."""

import base64
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from camera_processing.door_log_sync import copy_available, receiver, windows_log_server
from camera_processing.door_run_log import DoorRunLog


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.sources.append(dict(attrs)['src'])


class RunLogTests(unittest.TestCase):
    def test_host_copy_is_live_and_resumes_without_duplicates(self):
        self.assertIsNone(windows_log_server('http://127.0.0.1:18080'))
        self.assertEqual(windows_log_server('http://192.168.19.1:18080'), 'http://192.168.19.1:18081')
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with receiver(('127.0.0.1', 0), root/'windows') as host:
                worker = Thread(target=host.serve_forever, daemon=True)
                worker.start()
                try:
                    url = f'http://127.0.0.1:{host.server_port}'
                    journal = DoorRunLog(root/'ubuntu', mirror_url=url)
                    try:
                        journal.event('候选 <屏风>\n等待复核', b'jpeg')
                        copy_available(journal.path, url)
                        target = root/'windows'/journal.path.name
                        self.assertEqual(target.read_bytes(), journal.path.read_bytes())
                        journal.event('中断前的最后一帧')
                    finally:
                        journal.close()
                    self.assertEqual(target.read_bytes(), journal.path.read_bytes())
                    with target.open('ab') as output:
                        output.write(b'<p><time>partial')
                    copy_available(journal.path, url)
                    self.assertEqual(target.read_bytes(), journal.path.read_bytes())
                finally:
                    host.shutdown()
                    worker.join()

    def test_one_file_per_run_with_exact_frame_and_flushed_answer(self):
        with TemporaryDirectory() as root:
            directory = Path(root)/'door_nav_logs'
            first, second = DoorRunLog(directory), DoorRunLog(directory)
            try:
                self.assertNotEqual(first.path, second.path)
                frame = b'\xff\xd8\x00\xff\xd9'
                first.event('[门观测 1] 定向观察', frame)
                first.event("Qwen 原答='未见' <test>")
                html = first.path.read_text(encoding='utf-8')
                self.assertIn('未见', html)
                self.assertIn('&lt;test&gt;', html)
                images = Images()
                images.feed(html)
                src, = images.sources
                self.assertEqual(base64.b64decode(src.split(',')[1]), frame)
                self.assertNotIn('未见', second.path.read_text(encoding='utf-8'))
            finally:
                first.close()
                second.close()
