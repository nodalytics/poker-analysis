"""The trainer's window: a table you play against our bots, with a hint button and a verdict after each move."""

import math
import threading
import tkinter as tk
from tkinter import ttk

from advisor import trainer

FELT, RIM, INK, MUTED = "#1f6b47", "#14492f", "#f4f4f0", "#b9c9bf"
GRADE_COLOUR = {"agrees": "#1f6b47", "questionable": "#a86b00", "mistake": "#b5541c"}


def fmt_cards(cs):
    suits = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
    return " ".join(c[0].replace("T", "10") + suits[c[1]] for c in cs)


class TrainerTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.game = None
        self.busy = False
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Opponents").pack(side="left")
        self.picks = {}
        for key, label in trainer.BOTS.items():
            v = tk.BooleanVar(value=key in ("solverbot",))
            self.picks[key] = v
            ttk.Checkbutton(top, text=label.split(" (")[0], variable=v).pack(side="left", padx=2)
        opt = ttk.Frame(self)
        opt.pack(fill="x", pady=(6, 0))
        self.depth = tk.StringVar(value="deep")
        ttk.Radiobutton(opt, text="100bb", value="deep", variable=self.depth).pack(side="left")
        ttk.Radiobutton(opt, text="5-20bb", value="short", variable=self.depth).pack(side="left", padx=(4, 12))
        self.coach = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Verdict after each move", variable=self.coach).pack(side="left")
        ttk.Button(opt, text="New session", command=self.new_session).pack(side="right")
        self.canvas = tk.Canvas(self, height=300, bg="#f6f7f4", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.b_fold = ttk.Button(bar, text="Fold", command=lambda: self.hero("fold"))
        self.b_call = ttk.Button(bar, text="Check", command=lambda: self.hero("call"))
        self.b_raise = ttk.Button(bar, text="Raise to", command=lambda: self.hero("raise"))
        self.amount = tk.DoubleVar(value=0)
        self.e_amount = ttk.Spinbox(bar, from_=0, to=1000, increment=0.5, textvariable=self.amount, width=7)
        for w in (self.b_fold, self.b_call, self.b_raise, self.e_amount):
            w.pack(side="left", padx=3)
        ttk.Label(bar, text="bb").pack(side="left")
        for frac, lab in ((0.5, "1/2"), (0.66, "2/3"), (1.0, "pot"), (None, "all-in")):
            ttk.Button(bar, text=lab, width=5, command=lambda f=frac: self.size(f)).pack(side="left", padx=1)
        self.b_hint = ttk.Button(bar, text="Hint", command=self.hint)
        self.b_hint.pack(side="right")
        self.b_next = ttk.Button(bar, text="Next hand", command=self.next_hand)
        self.b_next.pack(side="right", padx=4)
        self.info = tk.Text(self, height=7, wrap="word", relief="flat")
        self.info.pack(fill="x", pady=(8, 0))
        for g, c in GRADE_COLOUR.items():
            self.info.tag_configure(g, foreground=c, font=("TkDefaultFont", 10, "bold"))
        self.set_buttons(False)
        self.say("Pick opponents and press New session. Play your hand; press Hint before acting to see the engine.")

    # ---- helpers ----
    def say(self, text, tag=None):
        self.info.insert("end", text + "\n", tag or ())
        self.info.see("end")

    def set_buttons(self, hero_turn):
        st = ["!disabled"] if hero_turn else ["disabled"]
        for w in (self.b_fold, self.b_call, self.b_raise, self.e_amount, self.b_hint):
            w.state(st)
        if hero_turn:
            lg = self.game.legal()
            self.b_call.configure(text=f"Call {lg['to_call'] / 2:g}bb" if lg["to_call"] else "Check")
            if not lg["can_fold"]:
                self.b_fold.state(["disabled"])
            if not lg["can_raise"]:
                self.b_raise.state(["disabled"])
                self.e_amount.state(["disabled"])
            else:
                self.b_raise.configure(text="Bet" if not lg["to_call"] and not lg["bet"] else "Raise to")
                self.amount.set(lg["min"] / 2)
        self.b_next.state(["!disabled"] if self.game and self.game.over() else ["disabled"])

    def size(self, frac):
        if not self.game or self.game.to_act() != trainer.HERO:
            return
        lg = self.game.legal()
        if not lg["can_raise"]:
            return
        if frac is None:
            to = lg["max"]
        else:
            to = lg["bet"] + lg["to_call"] + frac * (lg["pot"] + lg["to_call"])
        self.amount.set(round(max(lg["min"], min(lg["max"], to)) / 2, 1))

    # ---- flow ----
    def new_session(self):
        opp = [k for k, v in self.picks.items() if v.get()][:8]
        if not opp:
            self.say("Pick at least one opponent.")
            return
        self.game = trainer.TrainerGame(opp, self.depth.get())
        self.info.delete("1.0", "end")
        self.say(f"New session against {', '.join(trainer.BOTS[o] for o in opp)}.")
        self.next_hand()

    def next_hand(self):
        self.game.new_hand()
        self.shown_log = 0
        self.flush_log()
        self.run_bots()

    def flush_log(self):
        for line in self.game.history[self.shown_log :]:
            self.say(line)
        self.shown_log = len(self.game.history)

    def run_bots(self):
        """Let bots act one at a time (with a short pause) until it is the hero's turn or the hand ends."""
        self.draw()
        if self.game.over():
            self.flush_log()
            g = self.game
            dec = sum(g.decisions.values())
            self.say(
                f"Session: {g.hand_no} hands, you {g.net[trainer.HERO]:+.1f}bb"
                + (f"; your moves agreed with the engine {g.decisions['agrees']}/{dec}" if dec else "")
            )
            self.set_buttons(False)
            return
        if self.game.to_act() == trainer.HERO:
            self.flush_log()
            self.set_buttons(True)
            return
        self.set_buttons(False)

        def step():
            self.game.bot_act()
            self.after(0, lambda: (self.flush_log(), self.after(350, self.run_bots)))

        threading.Thread(target=step, daemon=True).start()

    def hint(self):
        if self.busy or self.game.to_act() != trainer.HERO:
            return
        adv = self.game.advice()
        m = adv["mix"]
        self.say(
            f"Hint ({adv['source']}): fold {m['fold']:.0%}, check/call {m['call']:.0%}, bet/raise {m['raise']:.0%}. "
            f"{adv['why']}"
        )

    def hero(self, kind):
        if self.busy or not self.game or self.game.to_act() != trainer.HERO:
            return
        to = None
        if kind == "raise":
            try:
                to = int(round(float(self.amount.get()) * 2))
            except (ValueError, tk.TclError):
                return
        self.set_buttons(False)
        v = self.game.hero_act(kind, to, coach=self.coach.get())
        self.flush_log()
        if v:
            m = v["advice"]["mix"]
            self.say(
                f"{v['grade'].upper()}: the engine plays this {v['pct']:.0%} "
                f"(fold {m['fold']:.0%}, check/call {m['call']:.0%}, bet/raise {m['raise']:.0%}; "
                f"{v['advice']['source']}). {v['advice']['why']}",
                v["grade"],
            )
        self.run_bots()

    # ---- drawing ----
    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or not self.game or not self.game.table:
            return
        cx, cy, rx, ry = w / 2, h / 2, w * 0.36, h * 0.3
        c.create_oval(cx - rx - 14, cy - ry - 14, cx + rx + 14, cy + ry + 14, fill=RIM, outline="")
        c.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, fill=FELT, outline="")
        g, s = self.game, self.game.table.state
        c.create_text(cx, cy - 16, text=fmt_cards(g.board()) or "", fill=INK, font=("TkDefaultFont", 18, "bold"))
        c.create_text(cx, cy + 18, text=f"Pot {s.total_pot_amount / 2:g}bb", fill=MUTED, font=("TkDefaultFont", 11))
        n = len(g.seats)
        hero_k = g.seats.index(trainer.HERO)
        showdown = set(g.showdown_names())
        acting = g.to_act()
        for k, name in enumerate(g.seats):
            ang = math.pi / 2 + 2 * math.pi * (k - hero_k) / n  # hero at the bottom
            x, y = cx + (rx + 46) * math.cos(ang), cy + (ry + 40) * math.sin(ang)
            folded = name in g.folded
            box = "#ffffff" if name == acting else "#e9ece8"
            c.create_rectangle(x - 78, y - 30, x + 78, y + 30, fill=box, outline="#c9cfc8")
            label = "You" if name == trainer.HERO else trainer.BOTS[name].split(" (")[0]
            btn = " (D)" if k == n - 1 else ""
            c.create_text(x, y - 16, text=label + btn, font=("TkDefaultFont", 10, "bold"))
            c.create_text(x, y - 1, text=f"{s.stacks[k] / 2:g}bb", font=("TkDefaultFont", 10))
            if name == trainer.HERO or name in showdown:
                cards = fmt_cards(g.cards(k).split())
            else:
                cards = "▮ ▮" if not folded else "folded"
            c.create_text(x, y + 15, text=cards, fill="#555" if folded else "#111", font=("TkDefaultFont", 11))
            if s.bets[k]:
                bx, by = cx + rx * 0.62 * math.cos(ang), cy + ry * 0.55 * math.sin(ang)
                c.create_text(bx, by, text=f"{s.bets[k] / 2:g}", fill="#ffe9a8", font=("TkDefaultFont", 11, "bold"))
