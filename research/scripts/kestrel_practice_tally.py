"""Tally a human's pre-flop raises and all-ins at Kestrel practice tables (how often every bot folded, chips won).

Usage: python3 kestrel_practice_tally.py SESSIONS_DIR KESTREL_DB  (copies of /data/sessions and /data/kestrel.db)
"""

import json
import glob
import os
import collections
import sys
import sqlite3

root = sys.argv[1]
MOD = {
    os.path.basename(o): m
    for m, o in sqlite3.connect(sys.argv[2]).execute("select model, out_dir from sessions where connector='practice'")
}
S = collections.defaultdict(lambda: collections.Counter())
R = collections.defaultdict(float)
for d in sorted(glob.glob(root + "/*-practice")):
    f = d + "/hands.jsonl"
    if not os.path.exists(f):
        continue
    rows = [json.loads(line) for line in open(f) if line.strip()]
    rows = [
        r
        for r in rows
        if r.get("type", "practice_hand") == "practice_hand" and any(x.get("you") for x in r.get("results", []))
    ]
    if not rows:
        continue
    model = MOD.get(os.path.basename(d), "?")
    for r in rows:
        pre = [x for x in r["log"] if x["street"] == 0]
        me = [x for x in pre if x["name"] == "You"]
        you = next(x for x in r["results"] if x["you"])
        key = model
        S[key]["hands"] += 1
        R[key, "all"] += you["result_bb"]
        aggr = [x for x in pre if x["action"].startswith(("raise", "all-in", "bet"))]
        if not me or not aggr or aggr[-1]["name"] != "You":
            continue
        kind = "jam" if aggr[-1]["action"].startswith("all-in") else "raise"
        after = pre[pre.index(aggr[-1]) + 1 :]
        resp = [x for x in after if x["name"] != "You"]
        called = any(x["action"].startswith(("call", "all-in")) for x in resp)
        S[key][kind] += 1
        S[key][kind + " all fold"] += not called
        R[key, kind] += you["result_bb"]
        if called:
            R[key, kind + " called"] += you["result_bb"]
            S[key][kind + " called"] += 1
for k, c in S.items():
    print(k, dict(c))
    print("   result bb:", {kk[1]: round(v, 1) for kk, v in R.items() if kk[0] == k})
