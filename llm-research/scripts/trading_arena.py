"""Trading ladder: models, learners, the regime-gated method from Minaei et al. (JRFM 2026, 19, 801), and controls,
all trading the same replayed market windows, after costs. Play money: a replay, never a live account.

Usage: uv run python scripts/trading_arena.py OUTDIR EPISODES MODEL [MODEL ...]

Each episode is a random window (EPISODE_HOURS of hourly decisions) of EURUSD (lab catalog, real bid/ask spread)
or BTC/USD (Coinbase through ccxt, fee per side). Everyone trades the same window; ranking is total R after costs
(R = the stop distance, so instruments are comparable), turned into pairwise Elo.

Entrants
  model#learn / model#frozen  as in the chess and poker ladder: the learner rewrites skill-trading.md and memory.md
                              after every episode, the frozen twin never sees notes
  qwen2.5vl:32b#paper         the model is given the paper's method each hour: its signal and its regime state
  paper_rule                  the paper's method with no model: EMA 20/50 trend, vetoed against the 200 SMA, taken
                              only when walk-forward K-means (k=3 on ATR/close, ROC-10, RSI-14; refit every 4 hours
                              on the previous 720 hours, scaler fitted on that window) puts the hour in the cluster
                              with the largest absolute ROC; stop 2 ATR, target 4 ATR
  paper_random_gate           the same signals with the K-means gate replaced by a coin with the same open rate
  random_trader, buy_hold, flat   controls
The paper reported results gross of costs; here every trade pays the spread (FX) or the fee (crypto).
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402  (reflect, notes, Elo helpers)
import trading_data as td  # noqa: E402

EPISODE_HOURS, WARMUP_HOURS, K = 72, 720, 24
CRYPTO_FEE = 0.001  # per side, a major venue's taker fee; Coinbase's retail tier is several times higher
PAPER_MODEL = "qwen2.5vl:32b"
RULES = ["paper_rule", "paper_random_gate", "random_trader", "buy_hold", "flat"]
OLLAMA = "http://127.0.0.1:11434"


# --------------------------------------------------------------------------------------------- features
def hourly(bars):
    h = (
        bars.set_index("time")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "spread": "median"})
        .dropna()
    )
    c = h["close"]
    h["ema20"], h["ema50"] = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    h["sma200"] = c.rolling(200).mean()
    d = c.diff()
    up, dn = (
        d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(),
        (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean(),
    )
    h["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([h["high"] - h["low"], (h["high"] - c.shift()).abs(), (h["low"] - c.shift()).abs()], axis=1).max(
        axis=1
    )
    h["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    h["atr_ratio"] = h["atr"] / c
    h["roc10"] = c.pct_change(10) * 100
    return h.dropna()


def paper_signal(row):
    if row["ema20"] > row["ema50"] and row["close"] > row["sma200"]:
        return "long"
    if row["ema20"] < row["ema50"] and row["close"] < row["sma200"]:
        return "short"
    return "flat"


def paper_regimes(h, start, end):
    """Favourable (True) or not for hours start..end-1, refitting every 4 hours on the previous 720, no look-ahead."""
    feats = ["atr_ratio", "roc10", "rsi"]
    out, model = {}, None
    for t in range(start, end):
        if model is None or (t - start) % 4 == 0:
            win = h.iloc[t - WARMUP_HOURS : t][feats].to_numpy()
            scaler = StandardScaler().fit(win)
            km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(scaler.transform(win))
            centres = scaler.inverse_transform(km.cluster_centers_)
            fav = int(np.argmax(np.abs(centres[:, 1])))
            model = (scaler, km, fav)
        scaler, km, fav = model
        out[t] = int(km.predict(scaler.transform(h.iloc[[t]][feats].to_numpy()))[0]) == fav
    return out


# --------------------------------------------------------------------------------------------- execution
class Book:
    """One position at a time; stop 2 ATR, target 4 ATR, managed on 15-minute bars; P&L in R after costs."""

    def __init__(self, cost_fn):
        self.pos, self.trades, self.cost_fn = None, [], cost_fn

    def open_r(self, price):
        p = self.pos
        return 0.0 if p is None else p["dir"] * (price - p["entry"]) / p["d"]

    def close(self, price, t, why, bar):
        p = self.pos
        cost = self.cost_fn(p["entry"], bar) + self.cost_fn(price, bar)
        r = p["dir"] * (price - p["entry"]) / p["d"] - cost / p["d"]
        self.trades.append(
            {
                "dir": "long" if p["dir"] > 0 else "short",
                "entry_time": str(p["t"]),
                "exit_time": str(t),
                "entry": p["entry"],
                "exit": price,
                "r": round(r, 3),
                "why": why,
            }
        )
        self.pos = None

    def act(self, action, price, t, atr, bar):
        want = {"long": 1, "short": -1}.get(action)
        if action == "flat" and self.pos:
            self.close(price, t, "flat", bar)
        elif want and (self.pos is None or self.pos["dir"] != want):
            if self.pos:
                self.close(price, t, "reverse", bar)
            self.pos = {"dir": want, "entry": price, "d": 2 * atr, "t": t}

    def manage(self, bar):
        p = self.pos
        if p is None:
            return
        stop, target = p["entry"] - p["dir"] * p["d"], p["entry"] + p["dir"] * 2 * p["d"]
        hit_stop = bar["low"] <= stop if p["dir"] > 0 else bar["high"] >= stop
        hit_target = bar["high"] >= target if p["dir"] > 0 else bar["low"] <= target
        if hit_stop:  # stop first when both are touched in one bar: the conservative assumption
            self.close(stop, bar["time"], "stop", bar)
        elif hit_target:
            self.close(target, bar["time"], "target", bar)


# --------------------------------------------------------------------------------------------- agents
def ask(model, prompt):
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.3, "num_predict": 60},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["message"]["content"]
    except urllib.error.HTTPError:
        return ""


def parse_action(text):
    low = (text or "").lower()
    for a in ("long", "short", "flat", "hold"):
        if f'"{a}"' in low or f"'{a}'" in low:
            return a
    for a in ("long", "short", "flat", "hold"):
        if a in low:
            return a
    return None


def observation(name, h, t, book, regime, signal):
    rows = h.iloc[t - 11 : t + 1]
    moves = ", ".join(f"{x:+.2f}%" for x in rows["close"].pct_change().dropna() * 100)
    r = h.iloc[t]
    pos = (
        "none"
        if book.pos is None
        else f"{'long' if book.pos['dir'] > 0 else 'short'}, open {book.open_r(r['close']):+.2f}R"
    )
    text = (
        f"You are trading {name} on hourly decisions in a replay, for play money. The hour just closed at "
        f"{r['close']:.5g}. Last 11 hourly moves: {moves}.\nEMA20 {r['ema20']:.5g}, EMA50 {r['ema50']:.5g}, "
        f"SMA200 {r['sma200']:.5g}, RSI14 {r['rsi']:.1f}, ATR14 {r['atr']:.5g}, ROC10 {r['roc10']:+.2f}%.\n"
        f"Your position: {pos}. A new position has a stop 2 ATR away and a target 4 ATR away; every trade pays "
        f"costs. R is profit in units of the stop distance.\n"
    )
    if regime is not None:
        text += (
            f"Method (Minaei et al. 2026): trade the EMA20/50 trend only with the 200 SMA, and only when a "
            f"walk-forward K-means regime model says the market is in its favourable state. Right now the "
            f"method's signal is {signal.upper()} and the regime is "
            f"{'FAVOURABLE (trading allowed)' if regime else 'ADVERSE (method says stay out)'}.\n"
        )
    return text + 'Reply with ONLY JSON: {"action": "long"|"short"|"flat"|"hold"}  (hold keeps the current position)'


def decide(entrant, h, t, book, regime, rng, gate_rate, notes):
    row = h.iloc[t]
    sig = paper_signal(row)
    if entrant == "flat":
        return "hold"
    if entrant == "buy_hold":
        return "long" if book.pos is None else "hold"
    if entrant == "random_trader":
        return rng.choice(["long", "short"]) if book.pos is None and rng.random() < 0.1 else "hold"
    if entrant == "paper_rule":
        return sig if regime and sig != "flat" else "hold"
    if entrant == "paper_random_gate":
        return sig if rng.random() < gate_rate and sig != "flat" else "hold"
    model = entrant.partition("#")[0]
    prompt = observation(INSTRUMENT_NAME[0], h, t, book, regime if entrant.endswith("#paper") else None, sig)
    note = notes.get(entrant, "")
    if note:
        prompt = f"Your own notes from earlier episodes (you wrote these):\n{note}\n\n" + prompt
    return parse_action(ask(model, prompt)) or "hold"


INSTRUMENT_NAME = ["?"]


# --------------------------------------------------------------------------------------------- episodes
def run_episode(outdir, ep, inst, h, bars15, start, entrants, ladder, log):
    INSTRUMENT_NAME[0] = inst["name"]
    end = start + EPISODE_HOURS
    regime = paper_regimes(h, start, end)
    gate_rate = sum(regime.values()) / len(regime)
    rng = random.Random(ep * 101)
    notes = {e: arena.notes_for(outdir, e, "trading") for e in entrants}
    books = {e: Book(inst["cost"]) for e in entrants}
    edir = os.path.join(outdir, "episodes", f"{ep:02d}")
    os.makedirs(edir, exist_ok=True)
    dlog = open(os.path.join(edir, "decisions.jsonl"), "w")
    for t in range(start, end):
        ts = h.index[t]
        nxt = bars15[(bars15["time"] > ts) & (bars15["time"] <= ts + pd.Timedelta("1h"))]
        if nxt.empty:
            continue
        price = float(nxt.iloc[0]["open"])
        for e in entrants:
            a = decide(e, h, t, books[e], regime[t], rng, gate_rate, notes)
            books[e].act(a, price, nxt.iloc[0]["time"], float(h.iloc[t]["atr"]), nxt.iloc[0])
            dlog.write(
                json.dumps(
                    {
                        "ep": ep,
                        "hour": str(ts),
                        "entrant": e,
                        "action": a,
                        "regime": regime[t],
                        "signal": paper_signal(h.iloc[t]),
                    }
                )
                + "\n"
            )
        for _, bar in nxt.iterrows():
            for e in entrants:
                books[e].manage(bar)
    last = bars15[bars15["time"] <= h.index[end - 1] + pd.Timedelta("1h")].iloc[-1]
    for e in entrants:
        if books[e].pos:
            books[e].close(float(last["close"]), last["time"], "episode end", last)
    totals = {e: round(sum(x["r"] for x in books[e].trades), 3) for e in entrants}
    with open(os.path.join(edir, "trades.json"), "w") as f:
        json.dump(
            {
                "instrument": inst["name"],
                "start": str(h.index[start]),
                "end": str(h.index[end - 1]),
                "paper_gate_open_share": round(gate_rate, 3),
                "totals": totals,
                "trades": {e: books[e].trades for e in entrants},
            },
            f,
            indent=1,
        )
    n, old = len(entrants), dict(ladder["trading"])
    for a in entrants:
        delta = 0.0
        for b in entrants:
            if a != b:
                s = 0.5 if abs(totals[a] - totals[b]) < 0.05 else float(totals[a] > totals[b])
                delta += (K / (n - 1)) * (s - arena.expected(old[a], old[b]))
        ladder["trading"][a] = old[a] + delta
    log(
        f"  episode {ep} {inst['name']} {h.index[start]:%Y-%m-%d %H:%M} +{EPISODE_HOURS}h, gate open "
        f"{gate_rate:.0%}: "
        + ", ".join(f"{arena.cs.short(e)} {totals[e]:+.2f}R" for e in sorted(entrants, key=lambda e: -totals[e]))
    )
    for e in entrants:
        if not e.endswith("#learn"):
            continue
        tr = books[e].trades
        wins = sum(1 for x in tr if x["r"] > 0)
        summary = (
            f"Episode on {inst['name']} ({EPISODE_HOURS} hourly decisions): {len(tr)} trades, total "
            f"{totals[e]:+.2f}R, {wins} winners. Exits: "
            + ", ".join(
                f"{k} {sum(1 for x in tr if x['why'] == k)}"
                for k in ("stop", "target", "flat", "reverse", "episode end")
            )
            + f". Best entrant this episode: {arena.cs.short(max(totals, key=totals.get))} {max(totals.values()):+.2f}R; "
            f"doing nothing (flat) scored 0R; buy and hold {totals.get('buy_hold', 0):+.2f}R."
        )
        skill, mem = arena.reflect(outdir, e, "trading", summary, ep)
        log(f"    {arena.cs.short(e)} rewrote skill-trading.md ({len(skill)} chars){'; memory: ' + mem if mem else ''}")


def main(outdir, episodes, models):
    os.makedirs(outdir, exist_ok=True)
    fx = td.catalog_bars("EURUSD.DERIV")
    btc = td.ccxt_bars("coinbase", "BTC/USD", days=90)
    insts = [
        {"name": "EURUSD", "bars": fx, "cost": lambda price, bar: float(bar["spread"]) / 2},
        {"name": "BTCUSD", "bars": btc, "cost": lambda price, bar: price * CRYPTO_FEE + float(bar["spread"]) / 2},
    ]
    for inst in insts:
        inst["h"] = hourly(inst["bars"])
    entrants = [f"{m}#{t}" for m in models for t in ("learn", "frozen")]
    entrants += [f"{PAPER_MODEL}#paper"] if PAPER_MODEL in models else []
    entrants += RULES
    path = os.path.join(outdir, "ladder.json")
    ladder = json.load(open(path)) if os.path.exists(path) else {"round": 0, "history": []}
    ladder.setdefault("trading", {})
    for e in entrants:
        ladder["trading"].setdefault(e, 1000.0)
    alog = open(os.path.join(outdir, "arena.log"), "a", buffering=1)
    pick = random.Random(7 + ladder["round"])
    for _ in range(episodes):
        ladder["round"] += 1
        ep = ladder["round"]
        inst = insts[(ep - 1) % len(insts)]
        h = inst["h"]
        start = pick.randrange(WARMUP_HOURS, len(h) - EPISODE_HOURS - 1)
        t0 = time.time()
        alog.write(f"\n=== Episode {ep} ===\n")
        run_episode(
            outdir, ep, inst, h, inst["bars"], start, entrants, ladder, alog.write and (lambda s: alog.write(s + "\n"))
        )
        ladder["history"].append({"round": ep, "trading": dict(ladder["trading"])})
        json.dump(ladder, open(path, "w"), indent=1)
        alog.write(f"  episode {ep} took {time.time() - t0:.0f}s\n")
    lines = ["# Trading ladder", "", "| Rank | Entrant | Elo |", "|---|---|---|"]
    for k, (e, r) in enumerate(sorted(ladder["trading"].items(), key=lambda x: -x[1]), 1):
        lines.append(f"| {k} | {arena.cs.short(e)} | {r:.0f} |")
    open(os.path.join(outdir, "ladder.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
