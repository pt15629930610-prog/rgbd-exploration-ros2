"""Mirror a four-door HTML journal from Ubuntu to the Windows model host."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import re
import subprocess
import threading
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


HTML_HEADER = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
               '<title>四门导航日志</title><body><h1>四门导航日志</h1>\n')
LOG_NAME = re.compile(r'/door-logs/(door_[0-9-]{10}_[0-9-]{8}_[a-z0-9_]{6,16}\.html)')
PARAGRAPH = re.compile(rb'<p><time>[0-9T.:+-]+</time> [^<>]*</p>\n')
IMAGE = re.compile(rb'<img alt="[^"<>&]*" src="data:image/jpeg;base64,[A-Za-z0-9+/=]+">\n')
PORT = 18081
UPLOAD_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>四门导航日志</title><h2>把已有的四门日志保存到 Windows</h2>
<p>在 Ubuntu 的主目录打开 door_nav_logs，选择要发送的 HTML 文件。</p>
<input type="file" accept=".html" aria-label="导航日志"><p role="status"></p>
<script>
const picker = document.querySelector('input');
const status = document.querySelector('[role=status]');
picker.onchange = async () => {
  const file = picker.files[0];
  if (!file) return;
  picker.disabled = true;
  try {
    let offset = 0;
    while (offset < file.size) {
      const bytes = new Uint8Array(await file.slice(offset, offset + 1048576).arrayBuffer());
      const end = bytes.lastIndexOf(10) + 1;
      if (!end) throw Error('日志行不完整');
      const response = await fetch('/door-logs/' + encodeURIComponent(file.name), {
        method: 'POST', body: bytes.subarray(0, end),
        headers: {'Content-Type': 'application/octet-stream', 'X-Log-Offset': String(offset)}
      });
      if (!response.ok && response.status !== 409) throw Error('HTTP ' + response.status);
      const next = Number(response.headers.get('X-Log-Offset'));
      if (next === offset || next < 0 || next > file.size) throw Error('日志长度不一致');
      offset = next;
    }
    status.textContent = 'Windows 已保存 ' + file.name;
  } catch (error) { status.textContent = '发送失败：' + error.message; }
  finally { picker.disabled = false; }
};
</script></html>"""


def receiver(address, directory):
    """Accept only complete journal records, with idempotent byte offsets."""
    directory.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        timeout = 4

        def do_GET(self):
            if self.path == '/':
                content, mime = UPLOAD_PAGE.encode(), 'text/html; charset=utf-8'
            elif self.path == '/health':
                content, mime = b'Windows door log receiver OK\n', 'text/plain'
            else:
                return self.send_error(404)
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            name = LOG_NAME.fullmatch(self.path)
            if not name or self.headers.get('Content-Type') != 'application/octet-stream':
                return self.send_error(404)
            try:
                offset = int(self.headers['X-Log-Offset'])
                length = int(self.headers['Content-Length'])
            except (KeyError, TypeError, ValueError):
                return self.send_error(400)
            if offset < 0 or not 0 < length <= 1024 * 1024:
                return self.send_error(400)
            body = self.rfile.read(length)
            lines = body.splitlines(keepends=True)
            if offset == 0 and (not lines or lines.pop(0) != HTML_HEADER.encode()):
                return self.send_error(400)
            if len(body) != length or not all(PARAGRAPH.fullmatch(line) or IMAGE.fullmatch(line)
                                               for line in lines):
                return self.send_error(400)
            path = directory / name[1]
            try:
                with path.open('a+b') as output:
                    size = output.tell()
                    if size:
                        output.seek(size - 1)
                        if output.read(1) != b'\n':
                            # A terminated writer can leave part of one record.
                            start = max(0, size - 1024 * 1024)
                            output.seek(start)
                            size = start + output.read().rfind(b'\n') + 1
                            output.truncate(size)
                    if offset == size:
                        output.write(body)
                        size += length
                        status = 204
                    else:
                        status = 409
            except OSError:
                return self.send_error(507)
            self.send_response(status)
            self.send_header('X-Log-Offset', str(size))
            self.end_headers()

        def log_message(self, format, *args):
            pass

    return HTTPServer(address, Handler)


def copy_available(path, server, offset=0):
    """Send flushed whole lines. On a lost acknowledgement resume at the receiver's size."""
    opener = build_opener(ProxyHandler({}))
    url = server.rstrip('/') + '/door-logs/' + path.name
    with path.open('rb') as source:
        while True:
            source.seek(offset)
            lines = source.readlines(96 * 1024)
            if lines and not lines[-1].endswith(b'\n'):
                lines.pop()
            if not lines:
                return offset
            body = b''.join(lines)
            request = Request(url, data=body, headers={'Content-Type': 'application/octet-stream',
                                                       'X-Log-Offset': str(offset)})
            try:
                with opener.open(request, timeout=1.5) as response:
                    offset = int(response.headers['X-Log-Offset'])
            except HTTPError as exc:
                with exc:
                    if exc.code != 409:
                        raise
                    remote = int(exc.headers['X-Log-Offset'])
                    if remote == offset or not 0 <= remote <= path.stat().st_size:
                        raise ValueError('Windows 日志长度与本地文件不符')
                    offset = remote


class LogMirror:
    def __init__(self, path, server):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.follow, args=(path, server), daemon=True)
        self.thread.start()

    def follow(self, path, server):
        offset, offline = 0, False
        while True:
            closing = self.stop.is_set()
            try:
                offset = copy_available(path, server, offset)
                if offline:
                    print(f'Windows 日志同步已恢复：{path.name}', flush=True)
                offline = False
            except (OSError, ValueError) as exc:
                if not offline:
                    print(f'Windows 日志暂未同步（{server}，{exc}），Ubuntu 保留原件。', flush=True)
                offline = True
            if closing:
                return
            self.stop.wait(3.0 if offline else .4)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)


def windows_log_server(vlm_server):
    host = urlsplit(vlm_server).hostname
    if host and host not in ('localhost', '127.0.0.1', '::1'):
        return f'http://[{host}]:{PORT}' if ':' in host else f'http://{host}:{PORT}'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    send = commands.add_parser('send')
    send.add_argument('server')
    send.add_argument('file', type=Path)
    listen = commands.add_parser('receive')
    listen.add_argument('--bind', required=True)
    listen.add_argument('--port', type=int, default=PORT)
    listen.add_argument('--directory', type=Path, default=Path(__file__).parent / 'door_nav_logs')
    listen.add_argument('--run', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'send':
        size = copy_available(args.file, args.server)
        print(f'Windows 已同步 {args.file.name}，共 {size} 字节。')
        return
    try:
        httpd = receiver((args.bind, args.port), args.directory)
    except OSError as exc:
        if not args.run:
            raise
        print(f'Windows 日志接收端未启动：{exc}', flush=True)
        raise SystemExit(subprocess.call(args.run))
    print(f'Windows 日志目录：{args.directory}（{args.bind}:{args.port}）', flush=True)
    with httpd:
        if args.run:
            worker = threading.Thread(target=httpd.serve_forever, daemon=True)
            worker.start()
            try:
                raise SystemExit(subprocess.call(args.run))
            finally:
                httpd.shutdown()
                worker.join()
        else:
            httpd.serve_forever()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
