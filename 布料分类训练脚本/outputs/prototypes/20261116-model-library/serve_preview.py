"""Local, temporary UI preview. Serves only this prototype, never model files."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import secrets

ROOT = Path(__file__).parent
KEY = secrets.token_urlsafe(20)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if parse_qs(url.query).get('key') != [KEY]:
            self.send_error(403)
            return
        if url.path not in ('/', '/index.html'):
            self.send_error(404)
            return
        body = (ROOT / 'index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
info = {'url': f'http://localhost:{server.server_port}/?key={KEY}', 'port': server.server_port}
(ROOT / 'server-info.json').write_text(json.dumps(info), encoding='utf-8')
print(json.dumps(info), flush=True)
server.serve_forever()
