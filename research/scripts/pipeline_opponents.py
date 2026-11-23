"""A network against tables of players that do not play like it, beside a baseline network on the same deals.

Matches against copies of nm25 (seqtest) only say how a network does against nm25; a change can win there and lose
to people who play differently. Here RUN and BASELINE each sit 1-vs-8 against eight copies of each rule player
(reg, tag, maniac, station, random: ninemax_selfplay.rule_probs), by duplicate on the same deals and in RUN's game.
Both sit on the same deals, stacks and payouts, so the difference is paired deal by deal and carries its own 95%
interval ("difference_ci"), much narrower than the two results' own intervals. The pipeline's "opponents" check passes
when RUN is no more than its tolerance below BASELINE against every one.

--styles K: also K generated opponents (and the named archetypes below). Each is the baseline network with its
tendencies pushed by random amounts, drawn from --style-seed so every run meets the same crowd: how loose and how
aggressive it is before the flop, how aggressive after it, how readily it folds to a bet, how often it shoves, how
much it bluffs with weak hands, and how random it is. A strong player with a human-shaped leak, many times over:
the pipeline's "styles" check needs RUN within its tolerance of BASELINE against the worst of them and on average.

Usage (lab, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/pipeline_opponents.py blueprint/nm44 --baseline blueprint/nm25 \
      --against reg,tag,maniac,station --out results/pipeline/nm44/opponents.json
"""

import argparse
import json
import math
import random

import torch
import torch.nn as nn

from ninemax_eval import load, setup
from ninemax_selfplay import duplicate


# the named archetypes: (pre_loose, pre_aggr, post_aggr, post_fold, shove, bluff, temp)
ARCHETYPES = {
    "nit": (-2.0, 0.0, -0.5, 1.0, -1.0, 0.0, 1.0),
    "station": (1.5, -1.5, -1.5, -3.0, -1.0, 0.0, 1.0),
    "maniac": (2.0, 2.0, 2.0, -1.0, 1.0, 0.8, 1.2),
    "shover": (1.0, 0.5, 0.0, 0.0, 3.0, 0.3, 1.0),
    "loose-passive": (2.0, -1.0, -1.0, 0.5, -1.0, 0.0, 1.3),
}
KEYS = ("pre_loose", "pre_aggr", "post_aggr", "post_fold", "shove", "bluff", "temp")


class Style(nn.Module):
    """base's policy with its tendencies pushed: logit offsets by action class (0 fold, 1 check/call, 2.. the bet and
    raise sizes, last all-in), by street, and a bluffing push that grows as the hand gets weaker (st: 0..1)."""

    def __init__(self, base, p):
        super().__init__()
        self.base, self.p = base, dict(zip(KEYS, p))

    def forward(self, feat, bk, street, st):
        lg, v = self.base(feat, bk, street, st)
        p, lg = self.p, lg.clone()
        pre = (street == 0).float()
        post = 1 - pre
        weak = (1 - st.float()).clamp(0, 1)
        lg[:, 0] += -p["pre_loose"] * pre + p["post_fold"] * post
        aggr = p["pre_aggr"] * pre + (p["post_aggr"] + 3 * p["bluff"] * weak) * post
        lg[:, 2:] += aggr[:, None]
        lg[:, -1] += p["shove"]
        return lg / p["temp"], v


def styles(k, seed):
    """The named archetypes, then k random styles from seed."""
    rng = random.Random(seed)
    out = dict(ARCHETYPES)
    for i in range(k):
        out[f"style{i + 1:02d}"] = (
            round(rng.uniform(-2, 2), 2), round(rng.uniform(-2, 2), 2), round(rng.uniform(-2, 2), 2),
            round(rng.uniform(-2.5, 2), 2), round(rng.uniform(-1.5, 2.5), 2), round(rng.uniform(0, 1), 2),
            round(rng.uniform(0.8, 1.5), 2),
        )  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline", default="blueprint/nm25")
    ap.add_argument("--against", default="reg,tag,maniac,station")
    ap.add_argument("--hands", type=int, default=20000, help="deals per match (x 9 seatings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="mtt")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp7.so")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--styles", type=int, default=0, help="also this many generated styles, and the archetypes")
    ap.add_argument("--style-seed", type=int, default=2026)
    ap.add_argument("--no-rules", action="store_true", help="only the styles, not --against")
    ap.add_argument("--human", default="", help="also the human-fitted styles in this file (human_styles.py fit)")
    a = ap.parse_args()
    dev = "cuda"
    dealer, menu, B = setup(a, dev)
    base, _ = load(a.baseline, B, dev)
    net, _ = load(a.run, B, dev)  # last: the game played is the run's
    out, seed0 = {}, dealer.seed
    opps = {} if a.no_rules else {r: r for r in a.against.split(",")}
    params = styles(a.styles, a.style_seed) if a.styles else {}
    if a.human:  # opponents shaped like real player types
        for st in json.load(open(a.human))["styles"]:
            params[st["id"]] = tuple(st["params"][k] for k in KEYS)
    opps.update({n: Style(base, p) for n, p in params.items()})
    for k, (rule, opp) in enumerate(opps.items()):
        row = {"style": dict(zip(KEYS, params[rule]))} if rule in params else {}
        per = {}
        for who, pol in (("run", net), ("baseline", base)):
            dealer.seed = seed0 + 1000 * k  # the dealer moves its seed on every deal: the same cards for both
            torch.manual_seed(a.seed + k)  # and the same stacks, payouts and rule-player draws
            m, lo, hi, per[who] = duplicate(dealer, menu, pol, opp, a.hands, dev, per_deal=True)
            row[who] = [round(float(x), 2) for x in (m, lo, hi)]
        d = per["run"] - per["baseline"]  # paired: the same deals, deal by deal (bb)
        md, sd = float(d.mean()) * 100, float(d.std()) / math.sqrt(len(d)) * 100
        row["difference"] = round(md, 2)
        row["difference_ci"] = [round(md - 1.96 * sd, 2), round(md + 1.96 * sd, 2)]
        out[rule] = row
        print(f"vs 8 x {rule:<8} run {row['run'][0]:+8.2f}  baseline {row['baseline'][0]:+8.2f}  "
              f"difference {row['difference']:+.2f} {row['difference_ci']} bb/100", flush=True)  # fmt: skip
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
