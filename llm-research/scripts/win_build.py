"""Build the Poker Advisor's Windows .exe inside the lab's Windows 11 VM (win11-gui), through Windows-MCP. Run on the lab.

Usage: python3 scripts/win_build.py probe            # tools, Python, internet in the VM
       python3 scripts/win_build.py build BUNDLE.zip OUT.exe
BUNDLE.zip holds llm-research's advisor/, scripts/, models and charts (made by the caller). Steps inside Windows:
install Python 3.12 if missing (python.org installer, per-user), upload the bundle in base64 chunks, create a venv,
pip install the advisor's requirements and PyInstaller, run advisor/packaging/build.py, run the .exe's --selftest,
and copy the .exe back in chunks. The MCP key is read from ~/windows11/oem/mcp.key and never printed;
~/windows11/in-use is held for the run so the 80-day rebuild never lands mid-build. Nothing else in the VM changes.
"""

import base64
import json
import os
import sys
import time
import urllib.request

MCP = "http://127.0.0.1:8765/mcp"
KEY = open(os.path.expanduser("~/windows11/oem/mcp.key")).read().strip()
INUSE = os.path.expanduser("~/windows11/in-use")
W = r"C:\advisor-build"
PY_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
PY = r"$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"  # expanded by PowerShell (always double-quoted)


class WinMCP:
    def __init__(self):
        self.sid, self.n = None, 0
        self.rpc(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "win-build", "version": "1"}},
        )
        self.rpc("notifications/initialized", {}, notify=True)
        self.tools = sorted(t["name"] for t in self.rpc("tools/list", {})["result"]["tools"])

    def rpc(self, method, params, notify=False, timeout=1800):
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notify:
            self.n += 1
            body["id"] = self.n
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer " + KEY,
        }
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(MCP, data=json.dumps(body).encode(), headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.sid = r.headers.get("Mcp-Session-Id", self.sid)
            text = r.read().decode()
        data = [ln[5:] for ln in text.splitlines() if ln.startswith("data:")]
        return json.loads(data[-1] if data else text) if (data or text.strip()) else {}

    def tool(self, name, args, timeout=1800):
        res = self.rpc("tools/call", {"name": name, "arguments": args}, timeout=timeout + 60)
        if "error" in res:
            raise RuntimeError(json.dumps(res["error"])[:500])
        parts = res.get("result", {}).get("content", [])
        return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")

    def ps(self, command, timeout=1800):
        """PowerShell's output only (the tool wraps it as 'Response: ... Status Code: n')."""
        text = self.tool("PowerShell", {"command": command, "timeout": int(timeout)}, timeout)
        body = text.split("Response:", 1)[-1]
        return body.rsplit("Status Code:", 1)[0].strip()

    def append(self, path, content):
        return self.tool("FileSystem", {"mode": "write", "path": path, "content": content, "append": True}, 120)


def probe(w):
    print("tools:", ", ".join(w.tools))
    print(
        w.ps(
            "$PSVersionTable.PSVersion.ToString(); py -0p 2>&1; where.exe python 2>&1; [Environment]::Is64BitOperatingSystem"
        )
    )
    print(
        w.ps(
            "try { (Invoke-WebRequest -UseBasicParsing -Uri https://pypi.org/simple/eval7/ -TimeoutSec 20).StatusCode } catch { 'no internet: ' + $_ }"
        )
    )
    print(w.ps("Get-PSDrive C | Select-Object Used,Free | Format-List"))


def upload(w, local, remote, chunk=2_000_000):
    """Base64 through FileSystem appends (no command-line length limit), then decoded inside Windows."""
    data = base64.b64encode(open(local, "rb").read()).decode()
    w.ps(
        f"New-Item -ItemType Directory -Force -Path '{os.path.dirname(remote)}' | Out-Null; Remove-Item -ErrorAction SilentlyContinue '{remote}.b64'; 'ok'"
    )
    for k in range(0, len(data), chunk):
        w.append(remote + ".b64", data[k : k + chunk])
    out = w.ps(
        f"[IO.File]::WriteAllBytes('{remote}', [Convert]::FromBase64String((Get-Content -Raw '{remote}.b64'))); (Get-Item '{remote}').Length",
        300,
    )
    assert out.splitlines()[-1].strip() == str(os.path.getsize(local)), out


def download(w, remote, local, chunk=1_500_000):
    size = int(w.ps(f"(Get-Item '{remote}').Length").splitlines()[-1])
    with open(local, "wb") as f:
        for off in range(0, size, chunk):
            b64 = w.ps(
                f"$fs=[IO.File]::OpenRead('{remote}'); $null=$fs.Seek({off},0); $b=New-Object byte[] {chunk}; "
                f"$r=$fs.Read($b,0,{chunk}); $fs.Close(); [Convert]::ToBase64String($b,0,$r)",
                300,
            )
            f.write(base64.b64decode("".join(b64.split())))
    assert os.path.getsize(local) == size, (os.path.getsize(local), size)


def build(w, bundle, out):
    log = []

    def step(name, cmd, timeout=1800):
        t = time.time()
        r = w.ps(cmd, timeout)
        log.append(f"== {name} ({time.time() - t:.0f}s)\n{r[-3000:]}")
        print(log[-1], flush=True)
        return r

    step(
        "python",
        f'if (-not (Test-Path "{PY}")) {{ $i="$env:TEMP\\py312.exe"; Invoke-WebRequest -UseBasicParsing -Uri {PY_URL} -OutFile $i; '
        f"Start-Process -Wait $i -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=0','Include_test=0' }}; & \"{PY}\" --version",
    )
    step(
        "clean",
        f"Remove-Item -Recurse -Force '{W}' -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force '{W}' | Out-Null; 'ok'",
    )
    upload(w, bundle, W + r"\bundle.zip")
    step(
        "unzip",
        f"Expand-Archive -Force '{W}\\bundle.zip' '{W}\\src'; Get-ChildItem '{W}\\src' | Select-Object -ExpandProperty Name",
    )
    step(
        "venv",
        f"& \"{PY}\" -m venv '{W}\\venv'; & '{W}\\venv\\Scripts\\python.exe' -m pip install -q --upgrade pip 2>&1 | Select-Object -Last 3; 'venv ok'",
    )
    step(
        "deps",
        f"& '{W}\\venv\\Scripts\\python.exe' -m pip install -q -r '{W}\\src\\advisor\\packaging\\requirements.txt' pyinstaller==6.22.3 2>&1 | Select-Object -Last 15; 'deps done'",
        3600,
    )
    step(
        "build",
        f"Set-Location '{W}\\src'; & '{W}\\venv\\Scripts\\python.exe' advisor\\packaging\\build.py 2>&1 | Select-Object -Last 12",
        3600,
    )
    step(
        "selftest",
        f'Remove-Item "$env:TEMP\\poker_advisor_selftest.txt" -ErrorAction SilentlyContinue; '
        f"$p=Start-Process -Wait -PassThru '{W}\\src\\dist\\PokerAdvisor.exe' -ArgumentList '--selftest'; "
        f"'exit ' + $p.ExitCode; Get-Content \"$env:TEMP\\poker_advisor_selftest.txt\" -ErrorAction SilentlyContinue",
        600,
    )
    download(w, W + r"\src\dist\PokerAdvisor.exe", out)
    print("downloaded", out, os.path.getsize(out), "bytes")


def main():
    open(INUSE, "w").write("win_build\n")
    try:
        w = WinMCP()
        if sys.argv[1] == "probe":
            probe(w)
        else:
            build(w, sys.argv[2], sys.argv[3])
    finally:
        os.remove(INUSE)


if __name__ == "__main__":
    main()
