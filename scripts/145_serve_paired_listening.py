"""Read-only loopback serving of one verified private listening pack.

No upload/review write endpoints, directory listings, LAN bind or external assets.
Only the exact playback/UI whitelist is reachable under a fresh secret URL prefix.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import mimetypes
from pathlib import Path
import re
import secrets
from urllib.parse import unquote, urlsplit

spec = importlib.util.spec_from_file_location("paired_listening_serve_pack", Path(__file__).with_name("144_export_paired_listening.py"))
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def byte_range(value, size):
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()) or size < 1:
        raise ValueError("Unsupported range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length < 1:
            raise ValueError("Empty suffix range")
        start, stop = max(0, size - length), size - 1
    else:
        start, stop = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or stop < start:
        raise ValueError("Range outside file")
    return start, stop, True


def create_server(out, port=62043, token=None):
    root = pack.guard_output(out)
    doc = pack.verify(root)
    files = pack.pack_files(doc)
    token = token or secrets.token_hex(24)
    if not re.fullmatch(r"[a-zA-Z0-9]{16,64}", token):
        raise ValueError("Invalid local URL secret")
    prefix = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not retain music paths or secret URLs in request logs.

        def do_HEAD(self):
            self.send_file(head=True)

        def do_GET(self):
            self.send_file(head=False)

        def send_file(self, head):
            path = unquote(urlsplit(self.path).path)
            name = path[len(prefix):] if path.startswith(prefix) else None
            if name not in files:
                self.send_error(404)
                return
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                self.send_error(404)
                return
            size = target.stat().st_size
            try:
                start, stop, partial = byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206 if partial else 200)
            media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if name.endswith(".js"):
                media_type = "application/javascript"
            self.send_header("Content-Type", media_type + ("; charset=utf-8" if not name.endswith(".wav") else ""))
            self.send_header("Content-Length", str(stop - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
            self.end_headers()
            if not head:
                try:
                    with target.open("rb") as stream:
                        stream.seek(start)
                        remaining = stop - start + 1
                        while remaining:
                            block = stream.read(min(65536, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.listening_url = f"http://127.0.0.1:{server.server_address[1]}{prefix}index.html"
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=pack.DEFAULT_OUT)
    ap.add_argument("--port", type=int, default=62043)
    args = ap.parse_args()
    pack.torch.set_num_threads(4)
    server = create_server(args.out, args.port)
    print("PAIRED_LISTENING_URL " + server.listening_url, flush=True)
    print("READ_ONLY LOOPBACK; no uploads; browser drafts are not training approval. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
