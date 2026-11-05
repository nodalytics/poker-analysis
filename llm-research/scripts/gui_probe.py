"""Step through a GUI task by hand with Windows-MCP to find a sequence that works. Run on the lab from ~/llm-research."""

import ast
import json
import re
import sys
import time

sys.path.insert(0, "scripts")
import gui_agent_test as g  # noqa: E402


def snap(w):
    raw = w.call("Snapshot", {})
    for parse in (json.loads, ast.literal_eval):
        try:
            v = parse(raw)
            return "\n".join(map(str, v)) if isinstance(v, list) else str(v)
        except (ValueError, SyntaxError):
            continue
    return raw.replace("\\n", "\n").replace('\\"', '"')


def find(s, pattern):
    m = re.search(pattern, s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def show(s, keys):
    for line in s.split("\n"):
        if re.search(keys, line):
            print("    " + line.strip()[:130])


task = sys.argv[1] if len(sys.argv) > 1 else "notepad"
w = g.WinMCP()
g.reset(w)
if task == "notepad":
    w.call("App", {"mode": "launch", "name": "Notepad"})
    time.sleep(3)
    s = snap(w)
    ed = find(s, r"\((\d+),(\d+)\) document \"Text editor\"")
    print("editor", ed, w.call("Type", {"loc": list(ed), "text": "hello from the model"})[:60])
    print(w.call("Shortcut", {"shortcut": "ctrl+s"})[:40])
    time.sleep(3)
    s = snap(w)
    print("after ctrl+s:")
    show(s, r"window \"|File name|edit \"|combo|button \"Save|button \"Cancel")
    box = find(s, r"\((\d+),(\d+)\) (?:edit|combo box) \"File name")
    print("file-name box", box)
    if box:
        print(
            w.call("Type", {"loc": list(box), "text": g.DESK + "\\gui_test.txt", "clear": True, "press_enter": True})[
                :80
            ]
        )
        time.sleep(3)
        s = snap(w)
        show(s, r"window \"")
    print("file:", w.ps(f"Get-Content '{g.DESK}\\gui_test.txt' -Raw -ErrorAction SilentlyContinue").strip()[:120])
elif task == "explorer":
    w.call("App", {"mode": "launch", "name": "File Explorer"})
    time.sleep(3)
    s = snap(w)
    bar = find(s, r"\((\d+),(\d+)\) edit \"Address Bar\"")
    print(
        "address bar", bar, w.call("Type", {"loc": list(bar), "text": g.DESK, "clear": True, "press_enter": True})[:70]
    )
    time.sleep(3)
    s = snap(w)
    show(s, r"window \"")
    print(w.call("Shortcut", {"shortcut": "ctrl+shift+n"})[:40])
    time.sleep(2)
    s = snap(w)
    print("after ctrl+shift+n:")
    show(s, r"New folder|edit \"|\[focused\]")
    box = find(s, r"\((\d+),(\d+)\) edit \"[^\"]*\"[^\n]*\[focused\]")
    print("rename box", box)
    if box:
        print(w.call("Type", {"loc": list(box), "text": "agent-test", "clear": True, "press_enter": True})[:80])
    time.sleep(2)
    print("folders:", w.ps(f"Get-ChildItem '{g.DESK}' -Directory | Select -Expand Name").strip()[:200])
