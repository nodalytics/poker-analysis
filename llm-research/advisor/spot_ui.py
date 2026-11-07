"""Spot study tab: type a situation, get the engine's answer, the opponents' likeliest holdings and your equity."""

import threading
import tkinter as tk
from tkinter import ttk

from advisor import spot

HELP = (
    "Action so far, in acting order:  f fold   x check   c call   r2.5 bet/raise to 2.5bb   a all-in   "
    "[Qs7h2c] next board.\n"
    "Example, 6-max, you in the big blind:  f f r2.5 f f c [Qs7h2c] x r3   "
    "(UTG folds, HJ folds, CO opens 2.5bb, BTN and SB fold, you call; flop; you check, CO bets 3bb)."
)


class SpotTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        g = ttk.Frame(self)
        g.pack(fill="x")
        self.n = tk.IntVar(value=6)
        self.stack = tk.DoubleVar(value=100)
        self.pos = tk.StringVar(value="BB")
        self.cards = tk.StringVar(value="AhKd")
        self.line = tk.StringVar(value="f f r2.5 f f c [Qs7h2c] x r3")
        self.solver = tk.BooleanVar(value=False)
        ttk.Label(g, text="Players").grid(row=0, column=0, sticky="w")
        sp = ttk.Spinbox(g, from_=2, to=9, textvariable=self.n, width=4, command=self.update_positions)
        sp.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(g, text="Stacks (bb)").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(g, textvariable=self.stack, width=7).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(g, text="Your position").grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.pos_box = ttk.Combobox(g, textvariable=self.pos, width=8, state="readonly")
        self.pos_box.grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(g, text="Your cards").grid(row=0, column=6, sticky="w", padx=(12, 0))
        ttk.Entry(g, textvariable=self.cards, width=8).grid(row=0, column=7, sticky="w", padx=4)
        ttk.Label(g, text="Action").grid(row=1, column=0, sticky="w", pady=(10, 0))
        e = ttk.Entry(g, textvariable=self.line)
        e.grid(row=1, column=1, columnspan=7, sticky="ew", padx=4, pady=(10, 0))
        e.bind("<Return>", lambda _: self.run())
        g.columnconfigure(7, weight=1)
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=8)
        self.go = ttk.Button(bar, text="Analyse", command=self.run)
        self.go.pack(side="left")
        ttk.Checkbutton(
            bar, text="Solve turns/rivers with TexasSolver (needs it installed)", variable=self.solver
        ).pack(side="left", padx=10)
        ttk.Label(self, text=HELP, foreground="#555", wraplength=900, justify="left").pack(fill="x")
        self.out = tk.Text(self, height=14, wrap="word", relief="flat")
        self.out.pack(fill="both", expand=True, pady=(8, 0))
        self.update_positions()

    def update_positions(self):
        names = spot.positions(int(self.n.get()))
        self.pos_box["values"] = names
        if self.pos.get() not in names:
            self.pos.set(names[-1])

    def run(self):
        self.go.state(["disabled"])
        self.out.delete("1.0", "end")
        self.out.insert("end", "Thinking...\n")
        args = (int(self.n.get()), float(self.stack.get()), self.pos.get(), self.cards.get().replace(" ", ""))

        def work():
            try:
                text = spot.describe(spot.analyse(*args, self.line.get(), use_solver=self.solver.get()))
            except Exception as ex:  # bad input is explained, not fatal
                text = f"Could not read the spot: {ex}"
            self.after(0, self.show, text)

        threading.Thread(target=work, daemon=True).start()

    def show(self, text):
        self.out.delete("1.0", "end")
        self.out.insert("end", text)
        self.go.state(["!disabled"])
