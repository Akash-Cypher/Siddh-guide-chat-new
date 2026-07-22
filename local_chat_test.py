"""
Local chat console for testing the LIVE production Siddh Guide chatbot.

Why this exists: the production backend only accepts the browser widget from
siddhantaknowledge.org (CORS) and needs a secret API key. This tiny local server
acts like the WordPress proxy — it keeps the key server-side and forwards your
messages to the live backend — so you can test the real production bot from your
PC without touching the live website.

Run it:
    python local_chat_test.py
Then open the URL it prints (http://localhost:8765) in your browser and chat.

Uses only the Python standard library — nothing to install.
"""

import json
import sys
import uuid
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- Point this at your live backend (already filled in) ---
BACKEND_URL = "https://nyrqbf2z3k.ap-south-1.awsapprunner.com/chat"
API_KEY = "siddh-guide-2026-8f3c9b2a71e4d5c6f9a0"
PORT = 8765

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Siddh Guide — Local Test Console</title>
<style>
  body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f4f6;color:#1a1a1a}
  header{background:#6b1f1f;color:#fff;padding:14px 18px;font-weight:600}
  header small{display:block;font-weight:400;opacity:.8;font-size:12px;margin-top:2px}
  #chat{max-width:720px;margin:0 auto;padding:16px;height:calc(100vh - 150px);overflow-y:auto}
  .msg{margin:10px 0;display:flex}
  .msg.user{justify-content:flex-end}
  .bubble{padding:10px 14px;border-radius:14px;max-width:80%;white-space:pre-wrap;line-height:1.4}
  .user .bubble{background:#6b1f1f;color:#fff;border-bottom-right-radius:4px}
  .bot .bubble{background:#fff;border:1px solid #e2e2e6;border-bottom-left-radius:4px}
  .cite{font-size:11px;color:#888;margin-top:4px}
  #bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #e2e2e6;padding:12px}
  #bar .wrap{max-width:720px;margin:0 auto;display:flex;gap:8px}
  #q{flex:1;padding:12px;border:1px solid #ccc;border-radius:10px;font-size:15px}
  #send{padding:12px 20px;background:#6b1f1f;color:#fff;border:0;border-radius:10px;cursor:pointer;font-size:15px}
  #send:disabled{opacity:.5}
</style></head>
<body>
<header>Siddh Guide — Local Test Console<small>Talking to LIVE production backend. Not on the live website.</small></header>
<div id="chat"></div>
<div id="bar"><div class="wrap">
  <input id="q" placeholder="Ask about courses, blogs, events, enrollment…" autofocus>
  <button id="send">Send</button>
</div></div>
<script>
const sid = "localtest-" + Math.random().toString(36).slice(2,10);
const chat = document.getElementById('chat'), q = document.getElementById('q'), send = document.getElementById('send');
function add(role, text, cites){
  const m=document.createElement('div'); m.className='msg '+role;
  const b=document.createElement('div'); b.className='bubble'; b.textContent=text; m.appendChild(b);
  if(cites && cites.length){ const c=document.createElement('div'); c.className='cite'; c.textContent='source: '+cites.join(', '); b.appendChild(c); }
  chat.appendChild(m); chat.scrollTop=chat.scrollHeight;
}
async function ask(){
  const text=q.value.trim(); if(!text) return;
  add('user', text); q.value=''; send.disabled=true;
  add('bot', '…');
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,session_id:sid})});
    const d=await r.json();
    chat.lastChild.remove();
    add('bot', d.answer||('[error] '+JSON.stringify(d)), d.citations);
  }catch(e){ chat.lastChild.remove(); add('bot','[network error] '+e); }
  send.disabled=false; q.focus();
}
send.onclick=ask; q.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
add('bot','Hi! I am the Siddh Guide assistant (live production). Ask me anything — e.g. "how many courses are there".');
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            payload.setdefault("session_id", "localtest-" + uuid.uuid4().hex[:8])
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                BACKEND_URL,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "x-api-key": API_KEY},
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                self._send(resp.status, resp.read())
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except Exception as e:  # noqa: BLE001
            self._send(502, json.dumps({"error": str(e)}))

    def log_message(self, *args):
        pass  # keep the console quiet


if __name__ == "__main__":
    # Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8 so
    # the banner never crashes the server before it starts.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n  Siddh Guide local test console")
    print(f"  Talking to: {BACKEND_URL}")
    print(f"\n  >> Open this in your browser:  http://localhost:{PORT}\n")
    print("  Press Ctrl+C to stop.\n")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
