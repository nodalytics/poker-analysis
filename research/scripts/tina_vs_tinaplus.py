import collections
import json
import os
import random
import sys

sys.path.insert(0, "scripts")
import tournament_env as te

T = sys.argv[1]
field = (
    [f"{b}#{k}" for b in ("tina", "tinaplus") for k in (1, 2, 3)]
    + [f"{b}#{k}" for b in ("bob", "max", "carl", "nora", "monte") for k in (1, 2)]
    + ["bob#3", "carl#3"]
)
te.main(T, 60, field)
rows = json.load(open(os.path.join(T, "results.json")))
grp = collections.defaultdict(lambda: {"prize": [], "itm": [], "finish": []})
for r in rows:
    for e, f in r["finish"].items():
        g = grp[e.split("#")[0]]
        g["prize"].append(r["prize"][e])
        g["itm"].append(f <= 3)
        g["finish"].append(f)
rng = random.Random(1)
out = [f"{len(rows)} tournaments, 18 players, prize pool 1800 (fair share 100 per entry)"]
for name, g in sorted(grp.items(), key=lambda x: -sum(x[1]["prize"]) / len(x[1]["prize"])):
    p = g["prize"]
    m = sum(p) / len(p)
    boot = sorted(sum(rng.choice(p) for _ in p) / len(p) for _ in range(2000))
    out.append(
        f"  {name:9} mean prize {m:6.1f}  95% CI [{boot[50]:6.1f}, {boot[1949]:6.1f}]  ITM {sum(g['itm']) / len(g['itm']):.0%}  mean finish {sum(g['finish']) / len(g['finish']):.1f}  (entries {len(p)})"
    )
open(os.path.join(T, "comparison.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
