"""Can a local model operate the Windows 11 desktop through Windows-MCP? Run on the lab.

Usage: python3 gui_agent_test.py [--harness] OUT.json MODEL [MODEL ...]   (tool-capable models)

--harness turns on harness.py: long Snapshot output is paged through `more_output` instead of cut at 3,500
characters, and the conversation is compacted before it overflows; tool descriptions are given in full.
Without it, results and descriptions are cut (the baseline). --skills puts gui-skills/windows-mcp.md in the system prompt.

Models get only Windows-MCP's GUI tools (App, Snapshot, Click, Type, Shortcut, Scroll, Move, Wait, WaitFor):
no PowerShell, FileSystem, Registry or Process, so a task cannot be scripted around the GUI. The grader uses
PowerShell itself, directly, to check each result inside Windows, and resets the desktop between models.
The key is read from ~/windows11/oem/mcp.key and never printed; ~/windows11/in-use is held for the run so the
80-day rebuild never lands mid-test.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

MCP = "http://127.0.0.1:8765/mcp"
OLLAMA = "http://127.0.0.1:11434"
KEY = open(os.path.expanduser("~/windows11/oem/mcp.key")).read().strip()
SKILLS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui-skills", "windows-mcp.md")
SKILLS = open(SKILLS_FILE).read() if os.path.exists(SKILLS_FILE) else ""
GUI_TOOLS = ["App", "Snapshot", "Click", "Type", "Shortcut", "Scroll", "Move", "Wait", "WaitFor"]
MAX_STEPS = 14
DESK = r"C:\Users\tester\Desktop"

SYSTEM = (
    "You operate a Windows 11 desktop through tools. Call Snapshot to see the screen as a list of UI elements with "
    "their coordinates, then Click, Type, Shortcut and App to act. Check with Snapshot after acting. When the task "
    "is done, reply in plain text with the answer or 'done'."
)


class WinMCP:
    def __init__(self):
        self.sid, self.n = None, 0
        init = self.rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "gui-agent-test", "version": "1"},
            },
        )
        self.server = init.get("result", {}).get("serverInfo", {})
        self.rpc("notifications/initialized", {}, notify=True)
        self.tools = {t["name"]: t for t in self.rpc("tools/list", {})["result"]["tools"]}

    def rpc(self, method, params, notify=False):
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
        with urllib.request.urlopen(req, timeout=120) as r:
            self.sid = r.headers.get("Mcp-Session-Id", self.sid)
            text = r.read().decode()
        data = [ln[5:] for ln in text.splitlines() if ln.startswith("data:")]
        return json.loads(data[-1] if data else text) if (data or text.strip()) else {}

    def call(self, name, args):
        res = self.rpc("tools/call", {"name": name, "arguments": args or {}})
        if "error" in res:
            return "ERROR: " + json.dumps(res["error"])[:500]
        parts = res.get("result", {}).get("content", [])
        return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")[:6000]

    def ps(self, command):
        return self.call("PowerShell", {"command": command})


class DesktopBroken(RuntimeError):
    pass


def reset(w):
    """Close the test apps and delete test files without touching the Windows shell, then prove the desktop works.

    File Explorer windows live inside the shell process (explorer.exe) on Windows 11, so they are closed through
    Shell.Application, never by killing explorer: killing it broke UI Automation ("The RPC server is unavailable")
    and invalidated a whole run on 2026-11-05.
    """
    w.ps(
        "(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.Quit() }; "
        "Get-Process notepad, CalculatorApp -ErrorAction SilentlyContinue | Stop-Process -Force "
        "-ErrorAction SilentlyContinue; "
        f"Remove-Item -Recurse -Force '{DESK}\\gui_test.txt', '{DESK}\\agent-test' -ErrorAction SilentlyContinue"
    )
    time.sleep(2)
    snap = w.call("Snapshot", {})
    if "Error capturing desktop state" in snap or snap.startswith("ERROR"):
        raise DesktopBroken(snap[:200])


TASKS = [
    {
        "id": "notepad_save",
        "prompt": f"Open Notepad, type exactly: hello from the model, then save it as {DESK}\\gui_test.txt",
        "check": lambda w, ans: "hello from the model" in w.ps(f"Get-Content '{DESK}\\gui_test.txt' -Raw"),
    },
    {
        "id": "calculator",
        "prompt": "Open the Calculator app, use it to work out 123 multiplied by 45, and tell me the result it shows.",
        "check": lambda w, ans: "5535" in (ans or "").replace(",", "")
        and "CalculatorApp" in w.ps("Get-Process CalculatorApp -ErrorAction SilentlyContinue | Select -Expand Name"),
    },
    {
        "id": "explorer_folder",
        "prompt": "Using File Explorer, create a new folder named agent-test on the Desktop.",
        "check": lambda w, ans: "True" in w.ps(f"Test-Path '{DESK}\\agent-test' -PathType Container"),
    },
]


def chat(model, messages, tools):
    body = {
        "model": model,
        "stream": False,
        "messages": messages,
        "tools": tools,
        "options": {"temperature": 0, "seed": 7, "num_predict": 400, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]


def run_task(model, w, task, use_harness=False, use_skills=False):
    tools = [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": " ".join(w.tools[n].get("description", "").split())[: None if use_harness else 600],
                "parameters": w.tools[n].get("inputSchema", {}),
            },
        }
        for n in GUI_TOOLS
        if n in w.tools
    ]
    if use_harness:
        tools.append(harness.MORE_OUTPUT_TOOL)
    chunks = harness.Chunks()
    system = SYSTEM + (
        "\n\nYour GUI skills notes (verified on this machine; follow them):\n" + SKILLS if use_skills else ""
    )
    ctx = harness.Context(model, [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}])
    log, answer = [], None
    for _ in range(MAX_STEPS):
        if use_harness:
            ctx.maybe_compact()
        try:
            msg = chat(model, ctx.messages, tools)
        except urllib.error.HTTPError as e:
            log.append({"server_error": f"{e.code}"})
            break
        ctx.add(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = msg.get("content", "")
            break
        for c in calls:
            fn = c["function"]
            name, args = fn["name"], fn.get("arguments") or {}
            if use_harness and name == "more_output":
                out = chunks.more(str(args.get("id", "")))
            elif name in GUI_TOOLS:
                out = w.call(name, args)
                out = chunks.page(out) if use_harness else out[:3500]  # baseline: cut, as in the first run
            else:
                out = f"ERROR: {name} is not available"
            log.append({"tool": name, "args": args, "result": out[:200]})
            ctx.add({"role": "tool", "content": out, "tool_name": name})
    time.sleep(1.5)
    try:
        ok = bool(task["check"](w, answer))
    except Exception as e:
        ok = False
        log.append({"check_error": str(e)[:200]})
    return {
        "task": task["id"],
        "passed": ok,
        "steps": len(log),
        "answer": answer,
        "log": log,
        "compactions": ctx.compactions,
        "pages_served": chunks.pages_served,
    }


if __name__ == "__main__":
    use_harness = "--harness" in sys.argv
    use_skills = "--skills" in sys.argv
    argv = [a for a in sys.argv[1:] if a not in ("--harness", "--skills")]
    out, models = argv[0], argv[1:]
    lock = os.path.expanduser("~/windows11/in-use")
    open(lock, "w").close()
    try:
        w = WinMCP()
        print(f"Windows-MCP {w.server.get('version')}: {len(w.tools)} tools; models get {GUI_TOOLS}", flush=True)
        results = []
        for model in models:
            r = {"model": model, "harness": use_harness, "skills": use_skills, "tasks": []}
            for t in TASKS:
                try:
                    reset(w)
                except DesktopBroken as e:
                    r["tasks"].append({"task": t["id"], "passed": None, "environment_error": str(e)})
                    print(f"{model} {t['id']}: ENVIRONMENT ERROR, not scored: {e}", flush=True)
                    raise SystemExit("desktop broken; stopping so no model is graded on it") from e
                t0 = time.time()
                res = run_task(model, w, t, use_harness, use_skills)
                res["seconds"] = round(time.time() - t0, 1)
                r["tasks"].append(res)
                print(
                    f"{model} {t['id']}: {'PASS' if res['passed'] else 'FAIL'} ({res['steps']} tool calls, "
                    f"{res['seconds']}s) answer={re.sub(r'\\s+', ' ', res['answer'] or '')[:80]!r}",
                    flush=True,
                )
            results.append(r)
            json.dump(results, open(out, "w"), indent=1)
        reset(w)
    finally:
        os.remove(lock)
