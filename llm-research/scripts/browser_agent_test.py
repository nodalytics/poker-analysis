"""Can a local model drive a browser? Same bridge as dc_agent_test, Playwright MCP tools.

Usage: python3 browser_agent_test.py OUT.json MODEL [MODEL ...]
The browser runs headless in a container with no network; the only site is one served inside it.
Checks read the site's own logs, so a model cannot pass by claiming success.
"""

import json
import sys
import time

import dc_agent_test as d

d.IMAGE = "browser-sandbox:0.0.83"
d.TOOLS = ["browser_navigate", "browser_snapshot", "browser_click", "browser_type"]
d.DESC_LIMIT = None  # full tool descriptions; the 300-character cut made the first run harder than necessary
SITE = "http://127.0.0.1:8000"


def last_code(ex):
    rows = [json.loads(x) for x in ex("cat /tmp/site/submissions.jsonl 2>/dev/null").splitlines() if x.strip()]
    return rows[-1]["code"] if rows else None


d.TASKS = [
    {
        "id": "read_price",
        "prompt": f"Open {SITE}/ in the browser and tell me the price of the Blue Mug.",
        "check": lambda ex, ans: "GET / " in ex("cat /tmp/site/access.log") and "7.50" in (ans or ""),
    },
    {
        "id": "click_link",
        "prompt": f"Open {SITE}/, click the Contact link, and tell me the main heading of the page it opens.",
        "check": lambda ex, ans: f"GET /contact referer={SITE}/" in ex("cat /tmp/site/access.log")
        and "contact us" in (ans or "").lower(),
    },
    {
        "id": "fill_form",
        "prompt": f"Go to {SITE}/contact, type Ada in the Name field and hello from the model in the Message field, then click Send.",
        "check": lambda ex, ans: any(
            r.get("name") == "Ada" and r.get("message") == "hello from the model"
            for r in (json.loads(x) for x in ex("cat /tmp/site/submissions.jsonl 2>/dev/null").splitlines() if x.strip())),
    },
    {
        "id": "read_result",
        "prompt": "Look at the page currently open in the browser and tell me the reference code it shows.",
        "check": lambda ex, ans: (c := last_code(ex)) is not None and c in (ans or "").upper(),
    },
]

if __name__ == "__main__":
    out, models = sys.argv[1], sys.argv[2:]
    results = []
    for i, model in enumerate(models):
        native = "tools" in d.caps(model)
        mcp = d.MCP(f"brsb-{i}-{int(time.time())}")
        try:
            r = {"model": model, "mode": "native tools" if native else "JSON text", "tasks": []}
            for t in d.TASKS:
                t0 = time.time()
                res = d.run_task(model, mcp, t, native)
                res["seconds"] = round(time.time() - t0, 1)
                r["tasks"].append(res)
                print(f"{model} [{r['mode']}] {t['id']}: {'PASS' if res['passed'] else 'FAIL'} "
                      f"({res['steps']} tool calls, {res['seconds']}s)", flush=True)
            results.append(r)
            with open(out, "w") as f:
                json.dump(results, f, indent=1)
        finally:
            mcp.close()
