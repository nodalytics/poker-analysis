"""Hand Sync: imports once, waits while hands are arriving, reviews when the session is over, reads GG zips."""

import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advisor import sync  # noqa: E402
from advisor.tests.test_handhistory import ACR, GG, STARS  # noqa: E402

sync.review.engine.pa.SOLVER = "/nonexistent"


def test_sync_flow(tmp_path):
    src, home = tmp_path / "hh", tmp_path / "home"
    src.mkdir()
    (src / "table1.txt").write_text(STARS + "\n\n" + ACR)
    s = sync.Sync(home=str(home), sources=[str(src)], idle_minutes=20, hero="Hero")
    r = s.step(now=time.time())  # files just changed: import, but do not review yet
    assert r == {"added": 2, "waiting": 2, "report": None}
    r = s.step(now=time.time() + 3600)  # an hour later the session is over: review
    assert r["added"] == 0 and r["waiting"] == 0 and r["report"] and os.path.exists(r["report"])
    assert os.path.exists(home / "trends.html") and len(s.history) == 1
    # a second instance reloads the state and never re-imports the same hands
    s2 = sync.Sync(home=str(home), sources=[str(src)], idle_minutes=20, hero="Hero")
    (src / "table1.txt").write_text(STARS + "\n\n" + ACR + "\n\n")  # touched, same hands
    assert s2.step(now=time.time())["added"] == 0


def test_gg_zip_in_downloads(tmp_path):
    dl, home = tmp_path / "Downloads", tmp_path / "home"
    dl.mkdir()
    with zipfile.ZipFile(dl / "GG20261102-handhistory.zip", "w") as z:
        z.writestr("GG20261102 - NLH 0.05-0.1.txt", GG)
    (dl / "holiday-photos.txt").write_text("not poker")
    s = sync.Sync(home=str(home), sources=[str(dl)], idle_minutes=0, hero="Hero")
    r = s.step(now=time.time() + 60)
    assert r["added"] == 1 and r["report"]
    assert any("GGPoker" in d for d in os.listdir(home / "library"))
