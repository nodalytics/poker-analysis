"""Tiny local test site. Logs every request and form submission under /tmp/site."""
import html, json, os, secrets, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = "/tmp/site"
os.makedirs(LOG, exist_ok=True)

HOME = """<!doctype html><title>Sandbox Shop</title><h1>Sandbox Shop</h1>
<table><tr><th>Product</th><th>Price</th></tr>
<tr><td>Red Kettle</td><td>$24.00</td></tr><tr><td>Blue Mug</td><td>$7.50</td></tr>
<tr><td>Green Teapot</td><td>$31.25</td></tr></table>
<p><a href="/contact">Contact</a></p>"""

CONTACT = """<!doctype html><title>Contact</title><h1>Contact us</h1>
<form method="post" action="/submit">
<label>Name <input name="name"></label><br>
<label>Message <textarea name="message"></textarea></label><br>
<button type="submit">Send</button></form>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        with open(f"{LOG}/access.log", "a") as f:
            f.write(f"GET {self.path} referer={self.headers.get('Referer', '')}\n")
        self.send(CONTACT if self.path.startswith("/contact") else HOME)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        form = {k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(n).decode()).items()}
        code = secrets.token_hex(3).upper()
        with open(f"{LOG}/submissions.jsonl", "a") as f:
            f.write(json.dumps({**form, "code": code}) + "\n")
        self.send(f"<!doctype html><title>Thanks</title><h1>Thanks, {html.escape(form.get('name', ''))}</h1>"
                  f"<p>Your reference code is {code}.</p>")


HTTPServer(("127.0.0.1", 8000), H).serve_forever()
