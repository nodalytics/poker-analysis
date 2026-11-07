"""Hand Sync tab: the folders it watches, Auto on/off, a log, and links to the latest report and the trends page."""

import os
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, ttk

from advisor import sync

POLL_MINUTES = 5


class SyncTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.s = sync.Sync()
        self.busy = False
        self.last_report = None
        ttk.Label(self, text="Folders with your hand histories (your own hands only):").pack(anchor="w")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=4)
        self.lst = tk.Listbox(row, height=5)
        self.lst.pack(side="left", fill="x", expand=True)
        btns = ttk.Frame(row)
        btns.pack(side="left", padx=6)
        ttk.Button(btns, text="Add...", command=self.add).pack(fill="x")
        ttk.Button(btns, text="Remove", command=self.remove).pack(fill="x", pady=4)
        self.refresh_list()
        opt = ttk.Frame(self)
        opt.pack(fill="x", pady=6)
        ttk.Label(opt, text="Review a session after").pack(side="left")
        self.idle = tk.DoubleVar(value=self.s.idle)
        ttk.Spinbox(opt, from_=5, to=240, increment=5, textvariable=self.idle, width=5).pack(side="left", padx=4)
        ttk.Label(opt, text="minutes without new hands").pack(side="left")
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text=f"Auto (check every {POLL_MINUTES} min)", variable=self.auto).pack(
            side="left", padx=14
        )
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.go = ttk.Button(bar, text="Sync now", command=self.run)
        self.go.pack(side="left")
        ttk.Button(bar, text="Open latest report", command=self.open_report).pack(side="left", padx=6)
        ttk.Button(bar, text="Open trends", command=self.open_trends).pack(side="left")
        self.out = tk.Text(self, height=12, wrap="word", relief="flat")
        self.out.pack(fill="both", expand=True, pady=(8, 0))
        self.say(
            f"Library: {self.s.lib}. {len(self.s.seen)} hands so far, {len(self.s.history)} sessions reviewed.\n"
            "Hands are reviewed only after a session ends, never while they are still arriving."
        )
        self.after(2000, self.tick)

    def say(self, text):
        self.out.insert("end", text + "\n")
        self.out.see("end")

    def refresh_list(self):
        self.lst.delete(0, "end")
        for p in self.s.sources:
            self.lst.insert("end", ("[found] " if os.path.isdir(p) else "[missing] ") + p)

    def add(self):
        d = filedialog.askdirectory()
        if d and d not in self.s.sources:
            self.s.sources.append(d)
            self.s.save()
            self.refresh_list()

    def remove(self):
        for k in reversed(self.lst.curselection()):
            del self.s.sources[k]
        self.s.save()
        self.refresh_list()

    def tick(self):
        if self.auto.get():
            self.run(quiet=True)
        self.after(POLL_MINUTES * 60 * 1000, self.tick)

    def run(self, quiet=False):
        if self.busy:
            return
        self.busy = True
        self.go.state(["disabled"])
        self.s.idle = float(self.idle.get())

        def work():
            try:
                r = self.s.step()
            except Exception as e:  # shown, not fatal
                r = {"error": str(e)}
            self.after(0, self.done, r, quiet)

        threading.Thread(target=work, daemon=True).start()

    def done(self, r, quiet):
        self.busy = False
        self.go.state(["!disabled"])
        if "error" in r:
            self.say(f"Sync failed: {r['error']}")
            return
        if r["report"]:
            self.last_report = r["report"]
            h = self.s.history[-1]
            self.say(
                f"Session reviewed: {h['hands']} hands, {(h['bb_per_100'] or 0):+.1f} bb/100, "
                f"{h['mistakes_per_100_decisions']} mistakes per 100 decisions. Report: {r['report']}"
            )
            webbrowser.open("file://" + os.path.abspath(r["report"]))
        elif r["added"] or not quiet:
            more = " Waiting for the session to end before reviewing." if r["waiting"] else ""
            self.say(f"Imported {r['added']} new hands; {r['waiting']} waiting.{more}")

    def open_report(self):
        p = self.last_report or (self.s.history[-1]["report"] if self.s.history else None)
        if p and os.path.exists(p):
            webbrowser.open("file://" + os.path.abspath(p))

    def open_trends(self):
        p = os.path.join(self.s.home, "trends.html")
        if os.path.exists(p):
            webbrowser.open("file://" + os.path.abspath(p))
        else:
            self.say("No sessions reviewed yet.")
