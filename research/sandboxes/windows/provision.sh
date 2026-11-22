#!/bin/sh
# Bring the Windows 11 GUI VM up end to end, on the lab, and prove Windows-MCP works.
#
#   sh provision.sh              start the VM if it is not running, then verify
#   sh provision.sh --reinstall  wipe the Windows disk and install from scratch (about 20 minutes)
#
# Lives next to run-win11.sh and oem/ (windows-up.sh copies them to ~/windows11 on the lab). It:
#   1. creates ~/windows11/credentials.env (VM login) and ~/windows11/oem/mcp.key if missing, mode 600;
#   2. stages oem/install.bat and oem/start-mcp.bat in ~/windows11/oem (copied into C:\OEM at install);
#   3. starts the container from run-win11.sh;
#   4. waits for a real RDP handshake, then for an authenticated MCP initialize on 127.0.0.1:8765;
#   5. lists Windows-MCP's tools, which is the proof the GUI-control path works.
# Secrets are never printed.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
W=$HOME/windows11
REINSTALL=0
[ "${1:-}" = "--reinstall" ] && REINSTALL=1

mkdir -p "$W/storage" "$W/oem"
umask 077
[ -s "$W/credentials.env" ] || printf "USERNAME=tester\nPASSWORD=%s\n" \
  "$(head -c 18 /dev/urandom | base64 | tr -d '/+=' | head -c 20)" > "$W/credentials.env"
[ -s "$W/oem/mcp.key" ] || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$W/oem/mcp.key"
umask 022
if [ "$HERE" != "$W" ]; then
  cp "$HERE/oem/install.bat" "$HERE/oem/start-mcp.bat" "$W/oem/"
  cp "$HERE/run-win11.sh" "$W/run-win11.sh"
fi

if docker ps -a --format '{{.Names}}' | grep -qx win11-gui; then
  if [ "$REINSTALL" = 1 ] || ! docker ps --format '{{.Names}}' | grep -qx win11-gui; then
    echo "stopping win11-gui (clean shutdown, up to 120s)"
    docker stop -t 120 win11-gui >/dev/null || true
    docker rm win11-gui >/dev/null
  fi
fi
if [ "$REINSTALL" = 1 ]; then
  echo "wiping the Windows disk; the ISO is kept if present"
  rm -f "$W/storage/data.img" "$W/storage/setup.img" "$W/storage/windows.vars" "$W/storage/windows.boot"
  date +%s > "$W/installed-at"   # read by rebuild-if-expiring.sh; the evaluation runs 90 days from here
fi
[ -s "$W/installed-at" ] || date +%s > "$W/installed-at"
docker ps --format '{{.Names}}' | grep -qx win11-gui || sh "$W/run-win11.sh" >/dev/null
echo "container up; waiting for Windows (a fresh install takes about 20 minutes)"

python3 - "$W/oem/mcp.key" <<'EOF'
import json, socket, sys, time, urllib.request

key = open(sys.argv[1]).read().strip()
deadline = time.time() + 3600

def rdp_up():
    try:
        s = socket.create_connection(("127.0.0.1", 3389), timeout=5)
        s.sendall(bytes.fromhex("030000130ee000000000000100080003000000"))
        r = s.recv(64)
        return r[:1] == b"\x03" and len(r) >= 11 and r[5] == 0xD0
    except OSError:
        return False

def mcp(method, params, sid=None, ident=1):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
         "Authorization": "Bearer " + key}
    if sid:
        h["Mcp-Session-Id"] = sid
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if ident is not None:
        body["id"] = ident
    req = urllib.request.Request("http://127.0.0.1:8765/mcp", data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=20) as r:
        text = r.read().decode()
        sid = r.headers.get("Mcp-Session-Id", sid)
    for line in text.splitlines():
        if line.startswith("data:"):
            text = line[5:]
    return (json.loads(text) if text.strip() else {}), sid

t0 = time.time()
while not rdp_up():
    if time.time() > deadline:
        sys.exit("RDP never came up")
    time.sleep(15)
print(f"RDP handshake after {time.time() - t0:.0f}s")
while True:
    try:
        init, sid = mcp("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "provision", "version": "1"}})
        break
    except OSError:
        if time.time() > deadline:
            sys.exit("Windows-MCP never answered on 127.0.0.1:8765")
        time.sleep(15)
info = init.get("result", {}).get("serverInfo", {})
print(f"Windows-MCP answered after {time.time() - t0:.0f}s: {info.get('name')} {info.get('version')}")
mcp("notifications/initialized", {}, sid, ident=None)
tools, _ = mcp("tools/list", {}, sid, ident=2)
names = [t["name"] for t in tools.get("result", {}).get("tools", [])]
print(f"{len(names)} tools: {', '.join(names)}")
if not names:
    sys.exit("no tools listed")
EOF
