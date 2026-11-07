"""Poker Advisor desktop app: session review (now), trainer and spot study (next).

Run: python -m advisor.app     (Windows, macOS or Linux; Tkinter ships with Python)
Session review reads finished hands from hand-history files and writes an HTML report that opens in the browser.
It never watches a table or reads the screen; review after you play.
"""

import datetime as dt
import os
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import review  # noqa: E402

# where clients commonly keep hand histories on Windows (the app only offers the ones that exist)
COMMON = [
    ("ACR / WPN", r"C:\ACR Poker\handHistory"),
    ("PokerStars", os.path.expandvars(r"%LOCALAPPDATA%\PokerStars\HandHistory")),
    ("Winamax", os.path.expandvars(r"%APPDATA%\winamax\documents\accounts")),
]
OUT_DIR = os.path.join(os.path.expanduser("~"), "PokerReview")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Poker Advisor")
        self.geometry("720x480")
        self.minsize(560, 400)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.review_tab(nb)
        for name in ("Trainer", "Spot study"):
            f = ttk.Frame(nb, padding=16)
            ttk.Label(f, text=f"{name}: coming next.").pack(anchor="w")
            nb.add(f, text=name)

    def review_tab(self, nb):
        f = ttk.Frame(nb, padding=16)
        nb.add(f, text="Session review")
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Hand histories").grid(row=0, column=0, sticky="w")
        found = [p for _, p in COMMON if os.path.isdir(p)]
        self.folder = tk.StringVar(value=found[0] if found else "")
        ttk.Entry(f, textvariable=self.folder).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Folder…", command=self.pick).grid(row=0, column=2)
        ttk.Label(f, text="Your screen name").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.hero = tk.StringVar()
        ttk.Entry(f, textvariable=self.hero).grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Label(f, text="(blank: detected from 'Dealt to')", foreground="#666").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        self.solver = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f, text="Use TexasSolver on turns and rivers (slower; needs the solver installed)", variable=self.solver
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.go = ttk.Button(f, text="Review", command=self.start)
        self.go.grid(row=3, column=0, sticky="w", pady=14)
        self.bar = ttk.Progressbar(f, mode="determinate")
        self.bar.grid(row=3, column=1, columnspan=2, sticky="ew", padx=6)
        self.out = tk.Text(f, height=12, wrap="word", relief="flat")
        self.out.grid(row=4, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(4, weight=1)
        self.say(
            "Supported: PokerStars, GGPoker (export from PokerCraft), ACR / WPN, Ignition / Bovada, Winamax, PHH.\n"
            "Reviews finished hands only. Use it after your session, not during play."
        )

    def say(self, text):
        self.out.insert("end", text + "\n")
        self.out.see("end")

    def pick(self):
        d = filedialog.askdirectory(initialdir=self.folder.get() or os.path.expanduser("~"))
        if d:
            self.folder.set(d)

    def start(self):
        folder = self.folder.get().strip()
        if not os.path.exists(folder):
            messagebox.showerror("Poker Advisor", "Choose the folder (or a file) with your hand histories.")
            return
        self.go.state(["disabled"])
        self.bar["value"] = 0
        self.say(f"Reviewing {folder} …")
        threading.Thread(target=self.work, args=(folder,), daemon=True).start()

    def progress(self, k, n):
        self.after(0, lambda: self.bar.configure(maximum=max(n, 1), value=k))

    def work(self, folder):
        try:
            rep = review.run([folder], self.hero.get().strip() or None, self.solver.get(), self.progress)
            os.makedirs(OUT_DIR, exist_ok=True)
            path = os.path.join(OUT_DIR, f"review-{dt.datetime.now():%Y%m%d-%H%M%S}.html")
            open(path, "w", encoding="utf-8").write(review.to_html(rep))
            self.after(0, self.done, rep, path)
        except Exception as e:  # shown to the user instead of a silent failure
            self.after(0, self.failed, e)

    def done(self, rep, path):
        g = rep["grades"]
        self.say(
            f"{rep['hands_reviewed']} hands for {rep['hero']} ({rep['hands_skipped']} skipped): "
            f"{rep['net_bb']:+.1f} bb ({(rep['bb_per_100'] or 0):+.1f} bb/100). "
            f"Decisions: {g.get('agrees', 0)} agree, {g.get('questionable', 0)} questionable, "
            f"{g.get('mistake', 0)} mistakes.\nReport: {path}"
        )
        self.go.state(["!disabled"])
        webbrowser.open("file://" + os.path.abspath(path))

    def failed(self, e):
        self.say(f"Failed: {e}")
        self.go.state(["!disabled"])


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
