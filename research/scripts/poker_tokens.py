"""The language of a poker hand: one tokenizer for every source (a poker foundation model, stage 0, 2026-11-27).

After Kronos (a tokenizer turning candlesticks into discrete tokens, then a decoder-only transformer pretrained on
them), the same idea for hold'em: a hand becomes a sequence of tokens from a small fixed vocabulary, whatever its
source (our networks' self-play, the HandHQ human hands, the Pluribus hands), so one model can be pretrained on all of
them by next-token prediction.

A hand (the intermediate form, `Hand`):
  source       "self", "human" or "pluribus"; game "tourney" or "cash"; ante_bb (a big-blind ante: 0 or 1)
  stacks_bb    every player's stack at the start, in big blinds, in seat order: small blind, big blind, ..., button
               (heads-up the small blind is the button)
  holes        per player its two cards (0..51, rank * 4 + suit, "23456789TJQKA" x "cdhs") or None when unknown
  streets      per street reached: the board cards it dealt and its actions (player, kind, x): kind fold, check,
               call, bet, raise or allin; x the chips put in beyond the bet before it, over the pot then
  shown        the players whose cards were shown (a showdown)
  results_bb   every player's result in big blinds (None when unknown)

The sequence:
  BOS SRC_<source> GAME_<game> N_<players> ANTE_<0|1>  (POS_<p> STK_<b>) x n
  [full layout: HOLE POS_<p> card card, for every player whose cards are known]
  PRE (POS_<p> <action>)*  [FLOP c c c (POS_<p> <action>)*]  [TURN c ...]  [RIVER c ...]
  [SHOW POS_<p> card card]*  [reveal layout: REVEAL (POS_<p> card card)*]  END (POS_<p> RES_<b>)*  EOS
with <action> FOLD, CHECK, CALL_<b>, BET_<b>, RAISE_<b> or ALLIN, b a size bucket (SIZE_EDGES). The "public" layout
leaves the hole cards out until they are shown (and, for self-play, lists every player's cards after the showdown:
REVEAL), so a model learns what the actions say about the cards; "full" puts them first.
"""

import bisect
import math
from dataclasses import dataclass, field

CARD_RANKS, CARD_SUITS = "23456789TJQKA", "cdhs"
NINE = ["SB", "BB", "UTG", "UTG1", "MP", "LJ", "HJ", "CO", "BTN"]
SIZE_EDGES = [0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]  # 13 buckets of x (pot fractions)
STACK_EDGES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 35, 50, 75, 100, 150, 250]  # 18 buckets, big blinds
RES_EDGES = [-100, -50, -25, -12, -6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6, 12, 25, 50, 100]  # 17 buckets, big blinds


def card_str(c):
    return CARD_RANKS[c // 4] + CARD_SUITS[c % 4]


def card_int(s):
    return CARD_RANKS.index(s[0].upper()) * 4 + CARD_SUITS.index(s[1].lower())


def positions(n):
    """Position names by seat for n players (seat 0 the small blind, the last the button), as Kestrel names them."""
    if n == 2:
        return ["SB", "BB"]
    return ["SB", "BB"] + NINE[9 - (n - 2) :]


def _vocab():
    v = ["PAD", "BOS", "EOS", "SEP", "END", "PRE", "FLOP", "TURN", "RIVER", "HOLE", "SHOW", "REVEAL"]
    v += ["SRC_self", "SRC_human", "SRC_pluribus", "GAME_tourney", "GAME_cash", "ANTE_0", "ANTE_1"]
    v += [f"N_{n}" for n in range(2, 10)] + [f"POS_{p}" for p in NINE]
    v += [f"STK_{b}" for b in range(len(STACK_EDGES) + 1)] + ["STK_U"]  # STK_U: not recorded
    v += ["C_" + card_str(c) for c in range(52)]
    v += ["FOLD", "CHECK", "ALLIN"]
    v += [f"{k}_{b}" for k in ("CALL", "BET", "RAISE") for b in range(len(SIZE_EDGES) + 1)]
    v += [f"RES_{b}" for b in range(len(RES_EDGES) + 1)]
    return v


VOCAB = _vocab()
ID = {t: k for k, t in enumerate(VOCAB)}


@dataclass
class Hand:
    source: str
    game: str
    ante_bb: float
    stacks_bb: list
    holes: list
    streets: list = field(default_factory=list)  # [{"board": [cards], "actions": [(player, kind, x)]}]
    shown: list = field(default_factory=list)
    results_bb: list | None = None


def _cards(cs):
    """A player's two cards as tokens, the higher card first (so a pair of cards has one spelling)."""
    return ["C_" + card_str(c) for c in sorted(cs, reverse=True)]


def _b(edges, x):
    return bisect.bisect_right(edges, x)


def encode(h, layout="public"):
    """The hand's token ids. layout "public" (cards only when shown; self-play adds REVEAL) or "full" (cards first)."""
    n = len(h.stacks_bb)
    pos = positions(n)
    t = ["BOS", f"SRC_{h.source}", f"GAME_{h.game}", f"N_{n}", f"ANTE_{1 if h.ante_bb > 0 else 0}"]
    for p in range(n):
        sb = h.stacks_bb[p]
        t += [f"POS_{pos[p]}", f"STK_{_b(STACK_EDGES, sb)}" if math.isfinite(sb) else "STK_U"]
    if layout == "full":
        for p in range(n):
            if h.holes[p] is not None:
                t += ["HOLE", f"POS_{pos[p]}"] + _cards(h.holes[p])
    for k, st in enumerate(h.streets):
        t.append(["PRE", "FLOP", "TURN", "RIVER"][k])
        t += ["C_" + card_str(c) for c in st["board"]]
        for p, kind, x in st["actions"]:
            t.append(f"POS_{pos[p]}")
            t.append(kind.upper() if kind in ("fold", "check", "allin") else f"{kind.upper()}_{_b(SIZE_EDGES, x)}")
    if layout == "public":
        for p in h.shown:
            if h.holes[p] is not None:
                t += ["SHOW", f"POS_{pos[p]}"] + _cards(h.holes[p])
        if all(x is not None for x in h.holes):  # every card known (self-play, Pluribus): what each player held
            t.append("REVEAL")
            for p in range(n):
                t += [f"POS_{pos[p]}"] + _cards(h.holes[p])
    t.append("END")
    if h.results_bb is not None:
        for p in range(n):
            t += [f"POS_{pos[p]}", f"RES_{_b(RES_EDGES, h.results_bb[p])}"]
    t.append("EOS")
    return [ID[x] for x in t]


def decode(ids):
    """The tokens as text, one line per section (for reading and tests)."""
    out, line = [], []
    for k in ids:
        tok = VOCAB[k]
        if tok in ("PRE", "FLOP", "TURN", "RIVER", "HOLE", "SHOW", "REVEAL", "END", "EOS") and line:
            out.append(" ".join(line))
            line = []
        line.append(tok)
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


# ---- HandHQ and Pluribus hands (PHH: github.com/uoftcprg/phh-dataset) ----


def from_phh(d, source):
    """A PHH hand (hand_histories.parse_phh) as a Hand, or None when it cannot be read (not no-limit hold'em,
    straddles, odd blinds). PHH players p1, p2, ... sit small blind, big blind, ..., button."""
    if d.get("variant") != "NT":
        return None
    raw = d.get("starting_stacks")
    if isinstance(raw, str):  # "[inf, inf, ...]": the stacks were not recorded (HandHQ): unknown, read as deep
        raw = [x.strip() for x in raw.strip("[]").split(",")]
    stacks = [float(x) for x in raw]
    if not all(x > 0 for x in stacks):
        return None
    n = len(stacks)
    blinds = [float(x) for x in d.get("blinds_or_straddles", [0] * n)]
    if n < 2 or n > 9 or len(blinds) < 2 or blinds[1] <= 0 or any(b != 0 for b in blinds[2:]):  # straddles, posts
        return None
    bb = blinds[1]
    antes = [float(x) for x in d.get("antes", [0] * n)]
    holes = [None] * n
    shown = []
    contrib = [blinds[p] + antes[p] for p in range(n)]  # chips in the pot, by player
    street_bet = [blinds[p] for p in range(n)]  # chips put in this street
    bet, streets, raised = bb, [{"board": [], "actions": []}], False
    for a in d.get("actions", []):
        f = a.split()
        if f[0] == "d":
            if f[1] == "dh":
                p = int(f[2][1:]) - 1
                if f[3] != "????" and len(f[3]) == 4:
                    holes[p] = [card_int(f[3][:2]), card_int(f[3][2:])]
            elif f[1] == "db":
                cs = f[2]
                streets.append({"board": [card_int(cs[k : k + 2]) for k in range(0, len(cs), 2)], "actions": []})
                street_bet = [0.0] * n
                bet, raised = 0.0, False
            continue
        p = int(f[0][1:]) - 1
        pot = sum(contrib)
        if f[1] == "f":
            streets[-1]["actions"].append((p, "fold", 0.0))
        elif f[1] == "cc":
            owe = min(bet - street_bet[p], stacks[p] - contrib[p])
            kind = "check" if owe <= 0 else ("allin" if contrib[p] + owe >= stacks[p] - 1e-9 else "call")
            streets[-1]["actions"].append((p, kind, max(owe, 0.0) / max(pot, 1e-9)))
            street_bet[p] += max(owe, 0.0)
            contrib[p] += max(owe, 0.0)
        elif f[1] == "cbr":
            to = float(f[2])
            add = to - street_bet[p]
            x = (to - bet) / max(pot, 1e-9)
            allin = contrib[p] + add >= stacks[p] - 1e-9
            kind = "allin" if allin else ("raise" if (bet > 0 or raised) else "bet")
            streets[-1]["actions"].append((p, kind, x))
            street_bet[p], contrib[p] = to, contrib[p] + add
            bet, raised = max(bet, to), True
        elif f[1] == "sm":
            if len(f) > 2 and len(f[2]) == 4 and "?" not in f[2]:  # "????": still hidden (an all-in runout)
                holes[p] = [card_int(f[2][:2]), card_int(f[2][2:])]
                if p not in shown:
                    shown.append(p)
    res = None
    if "finishing_stacks" in d:  # Pluribus
        res = [(float(d["finishing_stacks"][p]) - stacks[p]) / bb for p in range(n)]
    elif "winnings" in d:  # HandHQ: what each player collected, less what it put in
        res = [(float(d["winnings"][p]) - contrib[p]) / bb for p in range(n)]
    return Hand(source, "cash", 1.0 if any(x > 0 for x in antes) else 0.0, [s / bb for s in stacks], holes, streets,
                shown, res)  # fmt: skip


# ---- self-play (ninemax_engine.Tables with its action log) ----


def from_tables(T, g, cards, results_chips, bb=100, ante_bb=1.0, game="tourney"):
    """The finished hand at table g of T (Tables with its action log ev/nev) as a Hand: cards [2P + 5] (numpy, as
    Dealer.deal gives them), results_chips [P] (payoffs). Empty seats are left out."""
    P = T.P
    seats = [s for s in range(P) if int(T.stack[g, s]) > 0]
    idx = {s: k for k, s in enumerate(seats)}
    holes = [[int(cards[2 * s]), int(cards[2 * s + 1])] for s in seats]
    nb = {0: 0, 1: 3, 2: 4, 3: 5}
    streets = [{"board": [], "actions": []}]
    raised = {0: False}
    for k in range(int(T.nev[g])):
        s, st, kind, x = T.ev[g, k].tolist()
        st, kind = int(st), int(kind)
        while len(streets) <= st:  # a new street: its board cards
            m = len(streets)
            streets.append({"board": [int(cards[2 * P + c]) for c in range(nb[m - 1], nb[m])], "actions": []})
        if kind == 0:
            act = "fold"
        elif kind == 1:
            act = "check" if x <= 1e-9 else "call"
        elif kind == 3:
            act = "allin"
        else:
            act = "raise" if (st == 0 or raised.get(st, False)) else "bet"
            raised[st] = True
        streets[st]["actions"].append((idx[int(s)], act, float(x)))
    if bool(T.showdown[g]):  # an all-in before the river: the rest of the board is dealt too
        while len(streets) < 4:
            m = len(streets)
            streets.append({"board": [int(cards[2 * P + c]) for c in range(nb[m - 1], nb[m])], "actions": []})
    shown = [idx[s] for s in seats if bool(T.showdown[g]) and not bool(T.folded[g, s])]
    res = [float(results_chips[s]) / bb for s in seats]
    return Hand("self", game, ante_bb, [float(T.stack[g, s]) / bb for s in seats], holes, streets, shown, res)


def check_roundtrip():
    """The vocabulary covers every token encode can write, and decode reads it back (a smoke test)."""
    h = Hand("human", "cash", 0, [100, 50.5, 3], [[51, 50], None, None],
             [{"board": [], "actions": [(2, "fold", 0.0), (0, "raise", 2.0), (1, "call", 0.5)]},
              {"board": [0, 4, 8], "actions": [(1, "check", 0), (0, "bet", 0.66), (1, "fold", 0)]}], [], [1.5, -2.5, 0])  # fmt: skip
    for layout in ("public", "full"):
        ids = encode(h, layout)
        assert all(0 <= k < len(VOCAB) for k in ids) and VOCAB[ids[0]] == "BOS" and VOCAB[ids[-1]] == "EOS"
    return decode(encode(h, "full"))


assert len(VOCAB) == len(set(VOCAB)) and math.isfinite(len(VOCAB))
