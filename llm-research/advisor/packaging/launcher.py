"""Entry point of the packaged Poker Advisor (PyInstaller). Also works from source: python advisor/packaging/launcher.py

The advisor and scripts/ modules are shipped as plain source files inside the bundle, keeping the repository layout
(advisor/, scripts/, mlflop/, range_inference/, slumbot_range/, results/preflop/), because they find their models
and charts relative to their own location. The imports below exist so PyInstaller bundles every library those
modules use. --selftest answers one spot without opening a window (used to check a fresh build).
"""

import os
import sys

# libraries the shipped modules import (PyInstaller cannot see inside data files)
import argparse  # noqa: F401
import ast  # noqa: F401
import collections  # noqa: F401
import concurrent.futures  # noqa: F401
import copy  # noqa: F401
import datetime  # noqa: F401
import glob  # noqa: F401
import gzip  # noqa: F401
import html  # noqa: F401
import io  # noqa: F401
import itertools  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import random  # noqa: F401
import re  # noqa: F401
import subprocess  # noqa: F401
import tempfile  # noqa: F401
import threading  # noqa: F401
import tkinter  # noqa: F401
import tkinter.ttk  # noqa: F401
import urllib.request  # noqa: F401
import warnings  # noqa: F401
import webbrowser  # noqa: F401
import zipfile  # noqa: F401

import eval7  # noqa: F401
import joblib  # noqa: F401
import numpy  # noqa: F401
import pokerkit  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.neural_network  # noqa: F401

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, BASE)


def selftest():
    from advisor import spot

    spot.engine.pa.SOLVER = "/nonexistent"
    print(spot.describe(spot.analyse(6, 100, "BB", "AhKd", "f f r2.5 f f c [Qs7h2c] x r3")))
    print(spot.describe(spot.analyse(2, 15, "BTN", "Kh7c", "")))
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        from advisor import app

        app.main()
