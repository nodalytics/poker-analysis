# Models kept in the repo

Small, finished models go in git so they can be used without the lab. Large ones go on a GitHub Release, with
their checksums here. Full training state (optimizer, regrets, card buckets) stays on the lab.

Every file's sha256 is in [`manifest.json`](manifest.json), along with what it is, how long it trained and its results.

## 9-max networks (`ninemax/<run>/`)

- `avg_fp16.pt` holds the playing weights (the moving average) in half precision, about 4.5 MB.
- `config.json` is the run's config: input size, menu, stack mode and tournament settings.
- Exported with `scripts/ninemax_export.py`.
- Half precision plays the same as the original: `nm4`'s export against the original, 1-vs-8 by duplicate over
  20,000 deals, read −4.4 [−10.7, +1.8] and +1.1 [−5.4, +7.5].

Load one anywhere a run is accepted, by giving the file instead of the run directory:

```
PYTHONPATH=scripts python scripts/ninemax_eval.py cross models/ninemax/nm4/avg_fp16.pt blueprint/nm1
PYTHONPATH=scripts python scripts/ninemax_search.py models/ninemax/nm4/avg_fp16.pt --hands 2000
```

The card buckets the networks read (`blueprint/out`, 660 MB) are not here. They are regenerated with the
abstraction build in `blueprint/` (see `blueprint/README.md`).

| Model | Game | Why it is kept |
|---|---|---|
| `nm4` | cash, 100bb, NeuRD | The least exploitable cash network: a fresh 30-minute exploiter wins +44 [+39, +50]; real tables +30.1 |
| `nm11` (once its 24 hours are done) | cash, random 20–50bb stacks | Realistic depths |
| `nm10` (once its 24 hours are done) | tournament, ICM, 5–15bb and 20–50bb tables | The tournament network |
| `nm12` | tournament, ICM, 10–25bb, no open-limping, `mtt` menu, 24h | The first 10–25bb tournament network |
| `nm13`, `nm14` | tournament, 10–25bb, the tournament regular in the pool, 12h | Phase C, before the ante |
| `nm15`, `nm16` | tournament, 10–25bb, big-blind ante 1bb, 12h | The ante; `nm15` warm from `nm13`, `nm16` fresh |
| **`nm17`** | tournament, 10–25bb, ante 1bb, half the final tables 2–8 handed, 12h | **The most complete tournament network**: heads-up and short-handed play in band |

## Heads-up data pack (GitHub Release)

[`headsup-pack-run2`](https://github.com/nodalytics/poker-analysis/releases/tag/headsup-pack-run2):
- `headsup-pack-run2.tar.zst`, 29.6 MB (80 MB unpacked);
- the advisor's heads-up blueprint: the 8-bit average strategy of run 2, the flop and turn buckets, the river cut
  points, the tree, and `bp.dll`;
- it stays on a Release because a binary of this size in git history would grow every clone, for good.

```
cd research
gh release download headsup-pack-run2 --repo nodalytics/poker-analysis
zstd -dc headsup-pack-run2.tar.zst | tar -x -C blueprint/      # -> blueprint/pack/
cd blueprint/pack && sha256sum -c <(python3 -c "import json; m = json.load(open('../../models/manifest.json'))['headsup_pack']['files_sha256']; print('\n'.join(f'{v}  {k}' for k, v in m.items()))")
```
