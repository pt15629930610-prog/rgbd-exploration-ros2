"""A shareable run record is readable before the simulator stops."""

import base64
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from camera_processing.door_run_log import DoorRunLog


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.sources.append(dict(attrs)['src'])


class RunLogTests(unittest.TestCase):
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
