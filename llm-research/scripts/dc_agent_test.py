"""Can a local model drive Desktop Commander? Each model gets a fresh sealed container.

Usage: python3 dc_agent_test.py OUT.json MODEL [MODEL ...]

The container has no network, no host mounts, 1 CPU, 1GB RAM, no capabilities.
Models with native tool support use Ollama tool calls; the rest use a JSON text protocol.
Every task is checked against the container itself with docker exec, not the model's claim.
"""

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
IMAGE = "dc-sandbox:0.3.0"
TOOLS = ["start_process", "read_file", "write_file", "list_directory"]
MAX_STEPS = 8
WORK = "/home/agent/work"

TASKS = [
    {
        "id": "write_file",
        "prompt": f"Create a file {WORK}/hello.txt containing exactly the text: hello sandbox",
        "check": lambda ex, ans: ex(f"cat {WORK}/hello.txt").strip() == "hello sandbox",
    },
    {
        "id": "run_command",
        "prompt": "Find out how many CPU cores this machine reports by running a shell command, then tell me the number.",
        "check": lambda ex, ans: re.search(rf"\b{ex('nproc').strip()}\b", ans or "") is not None,
    },
    {
        "id": "script_and_run",
        "prompt": f"Write a Python script {WORK}/sum.py that prints the sum of the integers 1 to 100, run it with python3, and tell me what it printed.",
        "check": lambda ex, ans: ex(f"python3 {WORK}/sum.py 2>&1").strip() == "5050" and "5050" in (ans or ""),
    },
    {
        "id": "list_and_count",
        "prompt": f"List the files in {WORK} and tell me how many files there are.",
        "check": lambda ex, ans: re.search(rf"\b{ex(f'ls -1A {WORK} | wc -l').strip()}\b", ans or "") is not None,
    },
]


class MCP:
    def __init__(self, name):
        self.name = name
        self.p = subprocess.Popen(
            ["docker", "run", "-i", "--rm", "--name", name, "--network", "none",
             "--memory", "1g", "--cpus", "1", "--pids-limit", "256",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges", IMAGE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "dc-agent-test", "version": "1"}})
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.tools = [t for t in self.rpc("tools/list", {})["tools"] if t["name"] in TOOLS]

    def rpc(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("MCP server exited")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == self.n:
                if "error" in msg:
                    return {"isError": True, "content": [{"type": "text", "text": json.dumps(msg["error"])}]}
                return msg["result"]

    def call(self, name, args):
        if name not in TOOLS:
            return f"ERROR: unknown tool {name}"
        r = self.rpc("tools/call", {"name": name, "arguments": args or {}})
        text = "\n".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        return text[:2000]

    def exec(self, cmd):
        return subprocess.run(["docker", "exec", self.name, "sh", "-c", cmd],
                              capture_output=True, text=True, timeout=30).stdout

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def chat(model, messages, tools=None):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0, "seed": 7, "num_predict": 600, "num_ctx": 8192}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["message"]


def short(desc):
    return " ".join(desc.split())[:300]


def text_protocol_prompt(tools):
    lines = ["You control a Linux machine through tools. To use a tool, reply with ONLY one JSON object:",
             '{"tool": "<name>", "arguments": {...}}',
             'When the task is done, reply with ONLY: {"final": "<your answer>"}',
             "You will receive each tool's result before your next reply. Tools:"]
    for t in tools:
        lines.append(f"- {t['name']}: {short(t['description'])} Arguments schema: "
                     f"{json.dumps(t['inputSchema'].get('properties', {}))} required: {t['inputSchema'].get('required', [])}")
    return "\n".join(lines)


def first_json(text):
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def run_task(model, mcp, task, native):
    log, answer = [], None
    if native:
        tools = [{"type": "function", "function": {"name": t["name"], "description": short(t["description"]),
                                                   "parameters": t["inputSchema"]}} for t in mcp.tools]
        messages = [{"role": "system", "content": "You control a Linux machine through the provided tools. Use them to do the task, then answer briefly."},
                    {"role": "user", "content": task["prompt"]}]
    else:
        tools = None
        messages = [{"role": "system", "content": text_protocol_prompt(mcp.tools)},
                    {"role": "user", "content": task["prompt"]}]
    for _ in range(MAX_STEPS):
        try:
            msg = chat(model, messages, tools)
        except urllib.error.HTTPError as e:
            log.append({"server_error": f"{e.code} {e.read().decode(errors='replace')[:300]}"})
            break
        if native:
            calls = msg.get("tool_calls") or []
            messages.append(msg)
            if not calls:
                answer = msg.get("content", "")
                break
            for c in calls:
                fn = c["function"]
                res = mcp.call(fn["name"], fn.get("arguments"))
                log.append({"tool": fn["name"], "args": fn.get("arguments"), "result": res[:300]})
                messages.append({"role": "tool", "content": res, "tool_name": fn["name"]})
        else:
            text = msg.get("content", "")
            messages.append({"role": "assistant", "content": text})
            obj = first_json(text)
            if obj is None:
                log.append({"unparsed": text[:300]})
                messages.append({"role": "user", "content": "Reply with ONLY one JSON object as instructed."})
                continue
            if "final" in obj:
                answer = str(obj["final"])
                break
            res = mcp.call(obj.get("tool"), obj.get("arguments"))
            log.append({"tool": obj.get("tool"), "args": obj.get("arguments"), "result": res[:300]})
            messages.append({"role": "user", "content": f"Tool result:\n{res}"})
    try:
        ok = bool(task["check"](mcp.exec, answer))
    except Exception as e:
        ok = False
        log.append({"check_error": str(e)})
    return {"task": task["id"], "passed": ok, "steps": len(log), "answer": answer, "log": log}


def caps(model):
    req = urllib.request.Request(OLLAMA + "/api/show", data=json.dumps({"model": model}).encode())
    with urllib.request.urlopen(req) as r:
        return json.load(r).get("capabilities", [])


if __name__ == "__main__":
    out, models = sys.argv[1], sys.argv[2:]
    results = []
    for i, model in enumerate(models):
        native = "tools" in caps(model)
        mcp = MCP(f"dcsb-{i}-{int(time.time())}")
        try:
            r = {"model": model, "mode": "native tools" if native else "JSON text", "tasks": []}
            for t in TASKS:
                t0 = time.time()
                res = run_task(model, mcp, t, native)
                res["seconds"] = round(time.time() - t0, 1)
                r["tasks"].append(res)
                print(f"{model} [{r['mode']}] {t['id']}: {'PASS' if res['passed'] else 'FAIL'} "
                      f"({res['steps']} tool calls, {res['seconds']}s)", flush=True)
            results.append(r)
            with open(out, "w") as f:
                json.dump(results, f, indent=1)
        finally:
            mcp.close()
