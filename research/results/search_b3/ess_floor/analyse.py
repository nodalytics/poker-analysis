import json

import numpy as np

for o in ["net", "call", "fold", "raise"]:
    a = np.load(f"ess_{o}_0.npy").astype(np.float64)
    b = np.load(f"ess_{o}_0.02.npy").astype(np.float64)
    ja, jb = json.load(open(f"ess_{o}_0.json")), json.load(open(f"ess_{o}_0.02.json"))
    d = b - a
    se = d.std(ddof=1) / len(d) ** 0.5
    print(
        f"{o:5s} none {a.mean():+.2f} {ja['ci95']} floor {b.mean():+.2f}  diff {d.mean():+.3f} "
        f"[{d.mean() - 1.96 * se:+.3f}, {d.mean() + 1.96 * se:+.3f}]  corr {np.corrcoef(a, b)[0, 1]:.3f} "
        f"identical {(d == 0).mean():.1%}  fallback {ja['ess_fallback']}/{jb['ess_fallback']} "
        f"p10 {ja['ess_p10']}/{jb['ess_p10']} searched {ja['searched_decisions']}/{jb['searched_decisions']}"
    )
