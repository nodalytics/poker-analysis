# web

The 9-max watch page: `index.html`, one self-contained file (plain HTML, CSS and JavaScript, no build step, no
external libraries). It is served, with its data, by `scripts/ninemax_watch.py` on the lab:

```
PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_watch.py        # on the lab, from llm-research
ssh -N -L 8790:127.0.0.1:8790 -i lab.pem labuser@LAB_HOST      # on your machine
```

then open http://localhost:8790. Views: training charts and checkpoint hand replays (default), real tables and
live hands (tabs). Details: `docs/2026-11-09-nine-max-plan.md`, "Watching it learn and play".
