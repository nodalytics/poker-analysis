"""Build the Poker Advisor as a native app for the operating system this runs on.

Usage (from llm-research/):   uv run --with pyinstaller==6.22.3 python advisor/packaging/build.py
Output: dist/PokerAdvisor.exe (Windows), dist/PokerAdvisor.app (macOS), dist/PokerAdvisor (Linux).
PyInstaller cannot cross-compile: build each OS's app on that OS (locally, or the CI workflow
.github/workflows/advisor-build.yml, which builds all three).
Bundled: the advisor and scripts/ source files, the flop model, the hand-reading models and the pre-flop charts.
Not bundled: TexasSolver (optional; set TEXASSOLVER_BIN to use it on turns and rivers).
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # llm-research/
SEP = ";" if os.name == "nt" else ":"
DATA = [
    ("advisor", "advisor"),
    ("scripts", "scripts"),
    ("mlflop/mlp.joblib", "mlflop"),
    ("range_inference/action_model.joblib", "range_inference"),
    ("slumbot_range/action_model.joblib", "slumbot_range"),
    ("results/preflop", "results/preflop"),
]


def main():
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        "PokerAdvisor",
        "--distpath",
        os.path.join(ROOT, "dist"),
        "--workpath",
        os.path.join(ROOT, "build"),
        "--specpath",
        os.path.join(ROOT, "build"),
        "--collect-submodules",
        "sklearn",
        "--collect-submodules",
        "pokerkit",
        "--collect-data",
        "pokerkit",
        "--collect-all",
        "eval7",  # compiled submodules (xorshift_rand ...) that PyInstaller does not detect
    ]
    for heavy in ("matplotlib", "pandas", "pyarrow", "ccxt", "chess", "PIL", "IPython", "pytest"):
        args += ["--exclude-module", heavy]  # in the environment, never used by the advisor
    if sys.platform == "darwin":
        args += ["--windowed"]  # an .app bundle
    elif os.name == "nt":
        args += ["--onefile", "--windowed"]  # one .exe without a console window
    else:
        args += ["--onefile"]
    for src, dst in DATA:
        args += ["--add-data", f"{os.path.join(ROOT, src)}{SEP}{dst}"]
    args.append(os.path.join(ROOT, "advisor", "packaging", "launcher.py"))
    print(" ".join(args))
    subprocess.run(args, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
