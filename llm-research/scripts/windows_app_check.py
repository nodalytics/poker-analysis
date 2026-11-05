"""Install and launch a Windows app in the win11-gui VM, and record how it behaves. Harness only: no model drives it.

Usage (on the lab): python3 scripts/windows_app_check.py NAME URL [SHA256|-] OUT_DIR

Steps, all through Windows-MCP's PowerShell, as the grader does:
  1. download into C:\\OEM\\installers, record the SHA-256 (or check the published one)
  2. scan with Windows Defender; stop if it reports a threat
  3. identify the installer type (NSIS, Inno Setup, MSI, InstallShield, Squirrel) and install silently with its switch
  4. find the installed executable, launch it, wait for its window
  5. record window titles, the UI Automation element count, process memory and CPU, and a screenshot
It stops at the app's first screen: for real-money clients (ACR) it never signs in and never opens a table.
"""

import base64
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gui_agent_test as g  # noqa: E402

SILENT = {
    "nsis": "/S",
    "inno": "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-",
    "msi": "/qn",
    "installshield": "/s /v/qn",
    "squirrel": "--silent",
}


def ps(w, cmd):
    out = w.ps(cmd)
    m = re.search(r"Response:\s*(.*?)\s*Status Code:\s*(\d+)", out, flags=re.S)
    return (m.group(1).strip(), int(m.group(2))) if m else (out.strip(), -1)


def main(name, url, sha, out):
    os.makedirs(out, exist_ok=True)
    rep = {"name": name, "url": url}
    w = g.WinMCP()
    g.reset(w)
    exe = f"C:\\OEM\\installers\\{os.path.basename(url)}"
    t0 = time.time()
    ps(
        w,
        "New-Item -ItemType Directory -Force C:\\OEM\\installers | Out-Null; "
        f"if (-not (Test-Path '{exe}')) {{ Invoke-WebRequest -UseBasicParsing '{url}' -OutFile '{exe}' }}",
    )
    rep["download_seconds"] = round(time.time() - t0, 1)
    got, _ = ps(w, f"(Get-FileHash '{exe}' -Algorithm SHA256).Hash.ToLower()")
    rep["sha256"] = got
    rep["sha256_check"] = (
        "published match"
        if sha not in ("", "-") and got == sha.lower()
        else ("MISMATCH" if sha not in ("", "-") else "recorded (no published checksum)")
    )
    if rep["sha256_check"] == "MISMATCH":
        return finish(rep, out)
    ps(w, f"Start-MpScan -ScanType CustomScan -ScanPath '{exe}'")
    threats, _ = ps(w, "(Get-MpThreatDetection | Measure-Object).Count")
    rep["defender_threats"] = threats
    if threats not in ("0", ""):
        rep["stopped"] = "Defender reported a threat"
        return finish(rep, out)
    kind, _ = ps(
        w,
        f"$t = [IO.File]::ReadAllText('{exe}'); "
        "if ($t -match 'Nullsoft') {'nsis'} elseif ($t -match 'Inno Setup') {'inno'} "
        "elseif ($t -match 'InstallShield') {'installshield'} elseif ($t -match 'Squirrel') {'squirrel'} "
        "elseif ('" + exe + "' -like '*.msi') {'msi'} else {'unknown'}",
    )
    rep["installer_type"] = kind
    args = SILENT.get(kind, "/S")
    t0 = time.time()
    code, _ = ps(
        w,
        f"$p = Start-Process '{exe}' -ArgumentList '{args}' -PassThru; "
        "if ($p.WaitForExit(600000)) { $p.ExitCode } else { 'timeout' }",
    )
    rep["install_exit"], rep["install_seconds"] = code, round(time.time() - t0, 1)
    found, _ = ps(
        w,
        "Get-ChildItem 'C:\\Program Files','C:\\Program Files (x86)',$env:LOCALAPPDATA,$env:APPDATA "
        f"-Recurse -Depth 4 -Filter *.exe -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.Name -match '{name}' -and $_.Name -notmatch 'unins|setup|update|crash' }} | "
        "Select-Object -First 3 -Expand FullName",
    )
    rep["executables"] = [x for x in found.splitlines() if x.strip()]
    if not rep["executables"]:
        rep["stopped"] = "no installed executable found"
        return finish(rep, out)
    app = rep["executables"][0]
    t0 = time.time()
    ps(w, f"Start-Process '{app}'")
    time.sleep(45)
    rep["launch_wait_seconds"] = round(time.time() - t0, 1)
    procs, _ = ps(
        w,
        f"Get-Process | Where-Object {{ $_.Path -like '{os.path.dirname(app)}*' }} | "
        "Select-Object Name, @{n='MB';e={[int]($_.WorkingSet64/1MB)}}, CPU, MainWindowTitle | "
        "ConvertTo-Json -Compress",
    )
    rep["processes"] = procs
    snap = w.call("Snapshot", {})
    rep["ui_elements"] = len(re.findall(r"\(\d+,\d+\) ", snap))
    rep["windows"] = re.findall(r'window \\"([^\\"]+)\\"', snap)[:8]
    shot = w.rpc("tools/call", {"name": "Screenshot", "arguments": {}})
    for part in shot.get("result", {}).get("content", []):
        if part.get("type") == "image":
            with open(os.path.join(out, f"{name}-windows.png"), "wb") as f:
                f.write(base64.b64decode(part["data"]))
            rep["screenshot"] = f"{name}-windows.png"
    return finish(rep, out)


def finish(rep, out):
    with open(os.path.join(out, f"{rep['name']}-windows.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "processes"}, indent=1))
    return rep


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else "-",
        sys.argv[4] if len(sys.argv) > 4 else "compat",
    )
