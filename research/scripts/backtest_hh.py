"""Backtest model flows on real historical spots: Pluribus's own decisions, all hole cards known.

Usage: uv run python scripts/backtest_hh.py PHH_DIR OUTDIR N_SPOTS FLOW [FLOW ...]
FLOW is MODEL#plain (the model alone) or MODEL#advised (also told Monte Carlo equity against the live opponents and
the pot odds, from information the player had at the time), or 'pluribus' (scores Pluribus against itself: a check), or the controls 'always_call' and 'fold_to_bets'.

For each sampled spot the flow sees exactly what Pluribus saw (its cards, the board, the pot, the amount to call,
its stack, the betting so far) and chooses fold, check/call or bet/raise. Scored on:
  agreement   the same kind of action as Pluribus
  equity EV   chips won by the choice if the hand were settled now by equity against the hands the opponents really
              held (Monte Carlo over the rest of the board): fold 0, call/check equity x (pot + call) - call,
              bet/raise scored as a call. A lower bound for raises: fold equity and later betting are left out.
  regret      the flow's equity EV minus Pluribus's, per spot (negative means it gave up chips Pluribus took)
The real opponent cards are used only for scoring, never shown to the flow.
"""

import json
import os
import random
import re
import sys
import urllib.request

import eval7

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_histories as hh  # noqa: E402
import poker_agents as pa  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"


def spots(h):
    """Decision points with full context, from one hand."""
    names = h["players"]
    n = len(names)
    stacks = list(h["starting_stacks"])
    blinds = list(h.get("blinds_or_straddles", [0] * n)) + [0] * n
    holes, board, street, history = {}, [], "preflop", []
    bets = [blinds[i] for i in range(n)]
    folded = set()
    pot_prev = 0
    for a in h["actions"]:
        p = a.split()
        if p[0] == "d":
            if p[1] == "dh":
                holes[int(p[2][1:]) - 1] = p[3]
            elif p[1] == "db":
                board += [p[2][i : i + 2] for i in range(0, len(p[2]), 2)]
                street = {3: "flop", 4: "turn", 5: "river"}.get(len(board), street)
                pot_prev += sum(bets)
                stacks = [s - b for s, b in zip(stacks, bets)]
                bets = [0] * n
                history.append(f"[{street}: {''.join(board)}]")
            continue
        if not p[0].startswith("p") or p[1] not in ("f", "cc", "cbr"):
            continue
        i = int(p[0][1:]) - 1
        to_call = max(bets) - bets[i]
        yield {
            "player": names[i],
            "seat": i,
            "players": n,
            "street": street,
            "hole": holes.get(i),
            "board": list(board),
            "pot": pot_prev + sum(bets),
            "to_call": to_call,
            "stack": stacks[i] - bets[i],
            "history": list(history),
            "action": p[1],
            "opp_holes": [holes.get(j) for j in range(n) if j != i and j not in folded],
            "live_opponents": sum(1 for j in range(n) if j != i and j not in folded),
            # opponents who have put chips in this hand (blinds count): the ones a range should be drawn for
            "in_pot": sum(1 for j in range(n) if j != i and j not in folded and (bets[j] or pot_prev)),
        }
        if p[1] == "f":
            folded.add(i)
            history.append(f"{names[i]} folds")
        elif p[1] == "cc":
            bets[i] += to_call
            history.append(f"{names[i]} {'calls ' + str(to_call) if to_call else 'checks'}")
        else:
            bets[i] = int(float(p[2]))
            history.append(f"{names[i]} raises to {bets[i]}")


def equity_vs_known(hole, board, opp_holes, sims=3000, rng=random.Random(5)):
    hero = [eval7.Card(hole[:2]), eval7.Card(hole[2:])]
    opps = [[eval7.Card(o[:2]), eval7.Card(o[2:])] for o in opp_holes if o and "?" not in o]
    if not opps:
        return None
    used = {str(c) for c in hero + [x for o in opps for x in o]} | set(board)
    rest = [c for c in eval7.Deck().cards if str(c) not in used]
    b0 = [eval7.Card(c) for c in board]
    win = 0.0
    for _ in range(sims):
        run = b0 + rng.sample(rest, 5 - len(b0))
        me = eval7.evaluate(hero + run)
        best = max(eval7.evaluate(o + run) for o in opps)
        win += 1.0 if me > best else 0.5 if me == best else 0.0
    return win / sims


def equity_ev(kind, eq, pot, to_call):
    if kind == "fold":
        return 0.0
    return eq * (pot + to_call) - to_call  # check/call, and bet/raise scored as a call


def ask(model, prompt):
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 60},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["message"]["content"]
    except OSError:
        return ""


def kind_of(text):
    low = (text or "").lower()
    m = re.search(r'"action"\s*:\s*"(\w+)', low)
    a = m.group(1) if m else next((x for x in ("fold", "check", "call", "raise", "bet") if x in low), "fold")
    return {"check": "passive", "call": "passive", "raise": "aggressive", "bet": "aggressive"}.get(a, "fold")


def decide(flow, s):
    if flow == "pluribus":
        return {"f": "fold", "cc": "passive", "cbr": "aggressive"}[s["action"]]
    if flow == "always_call":  # control: never folds, never raises
        return "passive"
    if flow == "fold_to_bets":  # control: checks when free, folds to any bet
        return "fold" if s["to_call"] else "passive"
    model, _, mode = flow.partition("#")
    legal = ("fold, " if s["to_call"] else "") + ("call" if s["to_call"] else "check") + ", raise"
    prompt = (
        f"No-Limit Texas Hold'em, 6-max, blinds 50/100. Street: {s['street']}. Your cards: {s['hole']}. "
        f"Board: {''.join(s['board']) or '-'}. Pot: {s['pot']}. To call: {s['to_call']}. Your stack: {s['stack']}. "
        f"Opponents still in: {s['live_opponents']}.\nAction so far: {'; '.join(s['history'][-12:]) or 'none'}\n"
    )
    if mode == "advised":
        hero = [eval7.Card(s["hole"][:2]), eval7.Card(s["hole"][2:])]
        opp = max(1, s.get("in_pot", s["live_opponents"]))
        eq = pa.equity_vs_random(hero, [eval7.Card(c) for c in s["board"]], opp)
        odds = s["to_call"] / (s["pot"] + s["to_call"]) if s["to_call"] else 0
        prompt += f"Simulation: your equity vs {opp} random hand(s) in the pot is {eq:.0%}; calling needs {odds:.0%}.\n"
    prompt += f'Legal: {legal}. Reply with ONLY JSON: {{"action": "fold"|"check"|"call"|"raise"}}'
    k = kind_of(ask(model, prompt))
    return "passive" if k == "fold" and not s["to_call"] else k  # folding when checking is free is a check


def main(phh_dir, outdir, n_spots, flows):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(42)
    pool = []
    for _f, h in hh.hands(phh_dir):
        if "actions" not in h:
            continue
        for s in spots(h):
            if s["player"] == "Pluribus" and s["hole"] and s["live_opponents"] >= 1:
                pool.append(s)
    sample = rng.sample(pool, min(n_spots, len(pool)))
    for s in sample:
        s["eq_true"] = equity_vs_known(s["hole"], s["board"], s["opp_holes"])
    results = {}
    with open(os.path.join(outdir, "spots.jsonl"), "w") as out:
        for flow in flows:
            agree = regret = n_ev = 0.0
            for s in sample:
                k = decide(flow, s)
                ref = {"f": "fold", "cc": "passive", "cbr": "aggressive"}[s["action"]]
                agree += k == ref
                if s["eq_true"] is not None:
                    regret += equity_ev(k, s["eq_true"], s["pot"], s["to_call"]) - equity_ev(
                        ref, s["eq_true"], s["pot"], s["to_call"]
                    )
                    n_ev += 1
                out.write(
                    json.dumps(
                        {
                            "flow": flow,
                            "street": s["street"],
                            "ref": ref,
                            "choice": k,
                            "eq_true": s["eq_true"],
                            "pot": s["pot"],
                            "to_call": s["to_call"],
                        }
                    )
                    + "\n"
                )
            results[flow] = {
                "spots": len(sample),
                "agreement_with_pluribus": round(agree / len(sample), 3),
                "mean_regret_chips_vs_pluribus": round(regret / max(1, n_ev), 1),
            }
            print(flow, results[flow], flush=True)
    json.dump(
        {"source": phh_dir, "pool": len(pool), "results": results},
        open(os.path.join(outdir, "summary.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])
